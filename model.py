import math
import random
import einops
import torch
import torch.nn as nn
from torch.nn import init
from torchvision import ops
import torch.nn.functional as F
import numpy as np
from einops import rearrange, repeat


class OCCI(nn.Module):
  def __init__(self, slot_num, slot_size, Nc, Np, num_iterations, mlp_hidden_size, use_imagine, im_size):
    super().__init__()
    self.num_iterations = num_iterations
    self.mlp_hidden_size = mlp_hidden_size
    self.slot_num = slot_num
    self.slot_size = slot_size
    self.Nc = Nc
    self.Np = Np
    self.use_imagine = use_imagine
    self.im_size = im_size
    self.slt_attn = SlotAttention(self.num_iterations, self.slot_num, self.slot_size, self.mlp_hidden_size)
    
    # slot_attention version
    self.conv = nn.Sequential(
        nn.Conv2d(in_channels=1, out_channels=slot_size, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=slot_size, out_channels=slot_size, kernel_size=3, stride=1, padding=1),
        )
    
    self.ctrl = Controller(self.slot_num, self.slot_size, self.slt_attn)
    # self.exec = Executor(self.slot_size * 2, self.Nc, self.Np)
    # self.dec = Decoder(self.im_size, self.slot_size)
    
    # CNN version
    self.cnn = nn.Sequential(
        nn.Conv2d(in_channels=1, out_channels=self.mlp_hidden_size, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=self.mlp_hidden_size, out_channels=self.mlp_hidden_size, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=self.mlp_hidden_size, out_channels=self.mlp_hidden_size, kernel_size=3, stride=1, padding=1)
        )
    self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.mlp_hidden_size*3, nhead=3, batch_first=True)
    self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=3)
    
    self.exec = Executor(self.mlp_hidden_size * 3, self.Nc, self.Np)
    self.dec = Decoder(self.im_size, self.slot_size)

    self.softmax = nn.Softmax(dim=1)
    self.ce_loss = nn.CrossEntropyLoss()


  def forward(self, samples):
    batch_size = samples['input'].shape[0]
    
    # shape as [batch_size * io_num, 1, 20, 20]
    train_i = rearrange(samples['input'][:,None,:,:,:], 'b d i h w->(b i) d h w') 
    train_o = rearrange(samples['output'][:,None,:,:,:], 'b d i h w->(b i) d h w')
    query_i = rearrange(samples['query_i'][:,None,:,:,:], 'b d i h w->(b i) d h w')
    # shape as [batch_size, 1, 20, 20]
    query_o = samples['query_o']
    
    # CNN version
    diff = torch.sub(train_i, train_o, alpha=1)
    inp_cnn  = rearrange(self.cnn(train_i), 'bi d h w->bi d (h w)')
    out_cnn  = rearrange(self.cnn(train_o), 'bi d h w->bi d (h w)')
    diff_cnn = rearrange(self.cnn(diff), 'bi d h w->bi d (h w)')
    io_rep = torch.cat((inp_cnn, out_cnn, diff_cnn), dim=1)
    io_rep = rearrange(io_rep, 'bi d n->bi n d')
    
    # encoder io_representation in io_pair wise and mean average
    inst_embed = self.encoder(io_rep)
    inst_embed = rearrange(inst_embed, '(b i) n d->b i n d', b = batch_size)
    inst_embed = torch.mean(inst_embed, dim=1)
    inst_embed = torch.mean(inst_embed, dim=1)
       

    ''' slot_attention version
    # shape as [batch_size, slot_size, io_num, flatten_im_size]
    train_i = einops.rearrange(self.conv(train_i.float()), '(b i) d h w-> b d i (h w)', b = batch_size)
    train_o = einops.rearrange(self.conv(train_o.float()), '(b i) d h w-> b d i (h w)', b = batch_size)
    query_i = einops.rearrange(self.conv(query_i.float()), '(b i) d h w-> b d i (h w)', b = batch_size)
    
    inst_embed, slots_i, slots_o = self.ctrl(train_i, train_o)
    
    q_i = rearrange(query_i[:,:,0,:], 'b d n->b n d')
    H_query = self.slt_attn(q_i)
    '''
    
    q_i = rearrange(self.cnn(query_i), 'bi d h w->bi (h w) d')
    H_query = self.slt_attn(q_i)
    
    c, p = self.exec.selection(inst_embed, get_prob=False)
    # H_update is of shape [batch_size, num_slot, slot_size]
    H_update = self.exec.update(H_query, c, p)

    # pred_out shape is [b, out_channel, im_size, im_size]
    pred_out = None
    # H_update = H_update.sum(dim=1)
    
    H_update = H_update.reshape(batch_size, -1)
    pred_out = self.dec.sb_decode(H_update)
#    for i in range(self.slot_num):
#      pred = self.dec.sb_decode(H_update[:,i,:].float())
#      pred_out = pred if pred_out is None else torch.add(pred, pred_out)
#      
#    # cat zero background
#    zero_bg = torch.ones(batch_size, 1, self.im_size, self.im_size) * 0.35
#    pred_out = torch.cat((zero_bg, self.softmax(pred_out)*0.65), dim=1)  
#    pred_out = torch.log(pred_out + 1e-20)
    
    out_img = pred_out.permute(0,2,3,1).contiguous().reshape(-1,10)

    # calculate Loss of reconstruction
    query_o = query_o.flatten().type(torch.LongTensor)

    # L_rec = self.ce_loss(out_img, query_o)
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

  def __init__(self, num_iterations, slot_num, slot_size, mlp_hidden_size,
               epsilon=1e-8):
    """Builds the Slot Attention module.
    Args:
      num_iterations: Number of iterations.
      slot_num: Number of slots.
      slot_size: Dimensionality of slot feature vectors.
      mlp_hidden_size: Hidden layer size of MLP.
      epsilon: Offset for attention coefficients before normalization.
    """
    super().__init__()
    self.num_iterations = num_iterations
    self.slot_num = slot_num
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

  def forward(self, inputs, slot_num=None):
    # `inputs` has shape [batch_size, inputs_size].
    b, n, d, device, dtype = *inputs.shape, inputs.device, inputs.dtype
    n_s = slot_num if slot_num is not None else self.slot_num
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
    
    self.importn = ops.MLP(self.h_size, [self.h_size], norm_layer=nn.LayerNorm, activation_layer=nn.ReLU)
    self.contrib = ops.MLP(self.h_size, [self.h_size], norm_layer=nn.LayerNorm, activation_layer=nn.ReLU)
    
    self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.slot_size*2, nhead=4, batch_first=True)
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
      
      # for batch matrix mul
      dot = w * h
      _inst_embed = torch.sum(dot, dim=2)
      inst_embed = torch.tensor([])
      for i in range(_inst_embed.shape[1]):
        _inst = _inst_embed[:,i:i+1,:]
        # shape as [b, 1, d]
        _inst = self.encoder(_inst)
        inst_embed = torch.cat((inst_embed, _inst), dim=1)
      
      inst_embed = torch.mean(inst_embed, dim=1)
      
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
    
    ''' slt_attention version
    self.pres = ops.MLP(int(self.p * 3/2), [256, int(self.p * 1/2)], norm_layer=nn.LayerNorm, activation_layer=nn.ReLU)
    self.up   = ops.MLP(int(self.p * 3/2), [256, int(self.p * 1/2)], norm_layer=nn.LayerNorm, activation_layer=nn.ReLU)
    '''
    self.pres = ops.MLP(int(self.p * 4/3), [256, int(self.p * 1/3)], norm_layer=nn.LayerNorm, activation_layer=nn.ReLU)
    self.up   = ops.MLP(int(self.p * 4/3), [256, int(self.p * 1/3)], norm_layer=nn.LayerNorm, activation_layer=nn.ReLU)
    
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
    def __init__(self, im_size, slot_size):
      super().__init__()
      self.im_size = im_size
      
      x = torch.linspace(-1, 1, self.im_size)
      y = torch.linspace(-1, 1, self.im_size)
      self.x_grid, self.y_grid = torch.meshgrid(x, y)
      
      dec_convs = [nn.Conv2d(in_channels=slot_size*3+2, out_channels=slot_size,
                                   kernel_size=3, padding=1),
                   nn.Conv2d(in_channels=slot_size, out_channels=64,
                            kernel_size=3, padding=1)]
      self.dec_convs = nn.ModuleList(dec_convs)
      self.last_conv = nn.Conv2d(in_channels=64, out_channels=10,
                                       kernel_size=3, padding=1)
    
    def sb_decode(self, slots):
        batch_size = slots.shape[0]
        z = slots.view(slots.shape + (1, 1))
        
        # NxDx20x20
        z = z.expand(-1, -1, self.im_size, self.im_size)
        #print(self.x_grid)
        #print(self.x_grid.shape)
        #x = self.x_grid.reshape(self.im_size, self.im_size, 1)
        #y = self.y_grid.reshape(self.im_size, self.im_size, 1)
        #xy = np.concatenate((x,y), axis=-1)
        #xy_tiled = torch.from_numpy(np.tile(xy, (batch_size, 1, 1, 1)).astype(np.float32))
        #print(xy_tiled.shape)
        
        z = torch.cat((self.x_grid.expand(batch_size, 1, -1, -1),
                       self.y_grid.expand(batch_size, 1, -1, -1), z), dim=1)

        for module in self.dec_convs:
            z = F.relu(module(z))
        out_img = self.last_conv(z)
        
        return out_img
   