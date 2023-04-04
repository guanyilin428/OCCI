import math
import random
import einops
import torch
import torch.nn as nn
from torch.nn import init
from torchvision import ops
import torch.nn.functional as F


class OCCI(nn.Module):
  def __init__(self, num_slots, slot_size, Nc, Np, use_imagine, im_size):
    super().__init__()
    self.num_slots = num_slots
    self.slot_size = slot_size
    self.Nc = Nc
    self.Np = Np
    self.use_imagine = use_imagine
    self.im_size = im_size
    self.slt_attn = SlotAttention(5, self.num_slots, self.slot_size, 256).double()
    
    self.conv = nn.Sequential(
        nn.Conv2d(in_channels=1, out_channels=slot_size, kernel_size=3, stride=1, padding=1),
        nn.ReLU()
        )
    self.ctrl = Controller(self.num_slots, self.slot_size, self.slt_attn)
    self.exec = Executor(self.slot_size * 2, self.Nc, self.Np)
    self.dec = Decoder(self.im_size, self.num_slots)

    self.ce_loss = nn.CrossEntropyLoss()


  def forward(self, samples):
    train_i = samples['input'][:,None,:,:]    
    train_o = samples['output'][:,None,:,:]
    query_i = samples['query_i'][:,None,:,:]
    query_o = samples['query_o']

    batch_size = train_i.shape[0]
    
    # shape as [batch_size, slot_size, io_num, flatten_im_size]
    train_i = self.conv(train_i.float())
    train_o = self.conv(train_o.float())
    query_i = self.conv(query_i.float())
        
    inst_embed, slots_i, slots_o = self.ctrl(train_i, train_o)
    
    q_i = einops.rearrange(query_i[:,:,0,:], 'b d n->b n d')
    H_query = self.slt_attn(q_i)
    
    c, p = self.exec.selection(inst_embed, get_prob=False)
    # H_update is of shape [batch_size, num_slot, slot_size]
    # can be view as [batch_size, channels, height, width]
    H_update = self.exec.update(H_query, c, p)
    
    pred_out = self.dec.sb_decode(H_update.float())
    out_img = pred_out.permute(0,2,3,1).reshape(-1,10)

    # calculate Loss of reconstruction
    query_o = query_o.flatten().type(torch.LongTensor)

    L_rec = self.ce_loss(out_img, query_o)
    L_total = L_rec
    
    # imagination
    if self.use_imagine:
      ic = random.randint(0, self.Nc-1)
      ip = random.randint(0, self.Np-1)
      
      # randomly select condition and p
      c_sel = self.exec.Vc[:,ic,:]
      p_sel = self.exec.Vp[:,ip,:]
      
      im_train_o = self.exec.update(slots_i, c_sel, p_sel)
      
      z_im, _, _ = self.ctrl(train_i, im_train_o)
      c_prob, p_prob = self.exec.selection(z_im, get_prob=True)
      c_tar = torch.tensor(ic).expand(batch_size)
      p_tar = torch.tensor(ip).expand(batch_size)
      L_im = self.ce_loss(c_prob, c_tar) + self.ce_loss(p_prob, p_tar)

      L_total = L_im + L_rec
      
    return L_total, pred_out


class SlotAttention(nn.Module):
  """Slot Attention module."""

  def __init__(self, num_iterations, num_slots, slot_size, mlp_hidden_size,
               epsilon=1e-8):
    """Builds the Slot Attention module.
    Args:
      num_iterations: Number of iterations.
      num_slots: Number of slots.
      slot_size: Dimensionality of slot feature vectors.
      mlp_hidden_size: Hidden layer size of MLP.
      epsilon: Offset for attention coefficients before normalization.
    """
    super().__init__()
    self.num_iterations = num_iterations
    self.num_slots = num_slots
    self.slot_size = slot_size
    self.mlp_hidden_size = max(mlp_hidden_size, slot_size)
    self.epsilon = epsilon
    self.scale = self.slot_size ** -0.5
    
    self.slots_mu = nn.Parameter(torch.randn(1, 1, slot_size))
    self.slots_log_sigma = nn.Parameter(torch.zeros(1, 1, slot_size))
    init.xavier_uniform_(self.slots_log_sigma)
    
    self.to_q = nn.Linear(slot_size, slot_size)
    self.to_k = nn.Linear(slot_size, slot_size) 
    self.to_v = nn.Linear(slot_size, slot_size)

    self.gru = nn.GRUCell(slot_size, slot_size)
    self.mlp = nn.Sequential(
      nn.Linear(slot_size, mlp_hidden_size),
      nn.ReLU(inplace=True),
      nn.Linear(mlp_hidden_size, slot_size)
    )
    
    self.norm_input = nn.LayerNorm(slot_size)
    self.norm_slots = nn.LayerNorm(slot_size)
    self.norm_pre_ff = nn.LayerNorm(slot_size)

  def forward(self, inputs, num_slots=None):
    # `inputs` has shape [batch_size, inputs_size].
    b, n, d, device, dtype = *inputs.shape, inputs.device, inputs.dtype
    n_s = num_slots if num_slots is not None else self.num_slots
    mu = self.slots_mu.expand(b, n_s, -1)
    sigma = self.slots_log_sigma.exp().expand(b, n_s, -1)
    
    slots = mu + sigma * torch.randn(mu.shape, device=device, dtype = dtype)
    
    inputs = self.norm_input(inputs)
    k, v = self.to_k(inputs), self.to_v(inputs)

    for _ in range(self.num_iterations):
        slots_prev = slots

        slots = self.norm_slots(slots)
        q = self.to_q(slots)
        
        dots = torch.einsum('bid,bjd->bij', q, k) * self.scale
        attn = dots.softmax(dim=1) + self.epsilon

        attn = attn / attn.sum(dim=-1, keepdim=True)

        updates = torch.einsum('bjd,bij->bid', v, attn)

        slots = self.gru(
            updates.reshape(-1, d),
            slots_prev.reshape(-1, d)
        )

        slots = slots.reshape(b, -1, d)
        slots = slots + self.mlp(self.norm_pre_ff(slots))

    return slots


class Controller(nn.Module):
  def __init__(self, slot_num, slot_size, slot_module):
    super().__init__()
    self.slot_num = slot_num
    self.slot_size = slot_size
    self.slot_module = slot_module.float()
    self.h_size = 2 * self.slot_size
    
    # self.importn = ops.MLP(self.h_size * self.slot_num, [self.h_size * self.slot_num], norm_layer=nn.LayerNorm, activation_layer=nn.ReLU)
    # self.contrib = ops.MLP(self.h_size * self.slot_num, [self.h_size * self.slot_num], norm_layer=nn.LayerNorm, activation_layer=nn.ReLU)
    
    self.importn = ops.MLP(self.h_size, [self.h_size], norm_layer=nn.LayerNorm, activation_layer=nn.ReLU)
    self.contrib = ops.MLP(self.h_size, [self.h_size], norm_layer=nn.LayerNorm, activation_layer=nn.ReLU)
    
    self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.slot_size*2, nhead=8)
    self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=3)
    
  def forward(self, input, output):
      batch_size = input.shape[0]

      slots_i = torch.tensor([])
      slots_o = torch.tensor([])
      for i in range(input.shape[2]):
        inp = einops.rearrange(input[:,:,i,:], 'b d n->b n d')
        out = einops.rearrange(output[:,:,i,:], 'b d n->b n d')
        
        s_i = self.slot_module(inp.float())
        s_o = self.slot_module(out.float())
        # slots shape is [batch_size, io_num, num_slot, slot_size]
        slots_i = torch.cat((slots_i, s_i[:,None,:,:]), dim=1)
        slots_o = torch.cat((slots_o, s_o[:,None,:,:]), dim=1)
       
      # concatenate io slots for MLP
      # shape of [batch_size, num_io, num_slot, 2*slot_size]
      slot_pairs = torch.cat((slots_i, slots_o), -1).float()
      w = self.importn(slot_pairs)
      h = self.contrib(slot_pairs)
      # breakpoint()
      
      # w = self.importn(slot_pairs.reshape(batch_size, -1))
      # h = self.contrib(slot_pairs.reshape(batch_size, -1))
      
      # for batch matrix mul
      dot = w * h
      # dot = dot.reshape(batch_size, self.slot_num, -1)
      _inst_embed = torch.sum(dot, dim=2)
      _inst_embed = torch.mean(_inst_embed, dim=1)
      inst_embed = self.encoder(_inst_embed)
      
      return inst_embed, slots_i, slots_o


class Executor(nn.Module):
  def __init__(self, p, Nc, Np):
    super().__init__()
    self.p = p
    self.fc = ops.MLP(self.p, [self.p], norm_layer=nn.LayerNorm, activation_layer=nn.ReLU)
    self.fp = ops.MLP(self.p, [self.p], norm_layer=nn.LayerNorm, activation_layer=nn.ReLU)
    
    self.scale = math.sqrt(self.p)
    self.Nc = Nc
    self.Np = Np
    self.Kc = torch.randn(self.Nc, self.p, requires_grad=True)
    self.Vc = torch.randn(self.Nc, self.p, requires_grad=True)
    self.Kp = torch.randn(self.Np, self.p, requires_grad=True)
    self.Vp = torch.randn(self.Np, self.p, requires_grad=True)
    
    self.pres = ops.MLP(int(self.p * 3/2), [256, int(self.p * 1/2)], norm_layer=nn.LayerNorm, activation_layer=nn.ReLU)
    self.up   = ops.MLP(int(self.p * 3/2), [256, int(self.p * 1/2)], norm_layer=nn.LayerNorm, activation_layer=nn.ReLU)

    
  # select nn according to the instruction embedding
  def selection(self, inst_embed, get_prob):
    Qc = self.fc(inst_embed)
    Qp = self.fp(inst_embed)
    
    c_prob = torch.einsum('bi, ci->bc', Qc, self.Kc) / self.scale
    p_prob = torch.einsum('bi, pi->bp', Qp, self.Kp) / self.scale
    c = torch.einsum('bc,ci->bi', F.softmax(c_prob, dim=-1), self.Vc)
    p = torch.einsum('bp,pi->bi', F.softmax(p_prob, dim=-1), self.Vp)
    
    if get_prob:
      return c_prob, p_prob
    else: return c, p
  
  def update(self, slots, c, p):
    H_new = torch.tensor([])
    # fn_sig = nn.Sigmoid()
    for k in range(slots.size(dim=1)):
      h = slots[:,k,:]
      hc = torch.cat((h, c), dim=-1)
      hp = torch.cat((h, p), dim=-1)
      # h_new = h + fn_sig(self.pres(hc.float())) * self.up(hp.float())
      h_new = h + self.pres(hc.float()) * self.up(hp.float())
      H_new = torch.cat((H_new, h_new[:,None,:]), dim=1)
    return H_new


class Decoder(nn.Module):
    def __init__(self, im_size, slot_num):
      super().__init__()
      self.im_size = im_size
      
      x = torch.linspace(-1, 1, self.im_size)
      y = torch.linspace(-1, 1, self.im_size)
      self.x_grid, self.y_grid = torch.meshgrid(x, y)
      
      dec_convs = [nn.Conv2d(in_channels=slot_num+2, out_channels=64,
                                   kernel_size=3, padding=1),
                   nn.Conv2d(in_channels=64, out_channels=64,
                            kernel_size=3, padding=1)]
      self.dec_convs = nn.ModuleList(dec_convs)
      self.last_conv = nn.Conv2d(in_channels=64, out_channels=10,
                                       kernel_size=3, padding=1)
    
    def sb_decode(self, slots):
        batch_size = slots.shape[0]
        z = slots.view(slots.shape[0], slots.shape[1], self.im_size, self.im_size)
        
        z = torch.cat((self.x_grid.expand(batch_size, 1, -1, -1),
                       self.y_grid.expand(batch_size, 1, -1, -1), z), dim=1)

        for module in self.dec_convs:
            z = F.relu(module(z))
        out_img = self.last_conv(z)
        
        return out_img