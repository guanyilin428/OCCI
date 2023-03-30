import torch
import torch.nn as nn
import data_load
from pathlib import Path

from model import OCCI
# macro for slot_size(eq. input_size), num_slot
# slot_attn = slot_attention.SlotAttention(2,5,400,32)

loader = data_load.data_loader

model_path = Path('./trained_occi_200ep.pth')
model = OCCI(batch_size=8, num_slots=3, slot_size=data_load.const_len, Nc=26, Np=4, use_imagine=False)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(200):
  print('Epoch {:d}'.format(epoch + 1))
  # if epoch == 200:
  #   optimizer.lr = 0.0001
    # model.use_imagine = True
  for batch_idx, batched_samples in enumerate(loader):
      L_rec, _ = model(batched_samples, batch_idx)
      
      optimizer.zero_grad()
      L_rec.backward()
      optimizer.step()
      if batch_idx % 10 == 9:
        print(f"loss: {L_rec:>7f}")

torch.save(model.state_dict(), model_path)
