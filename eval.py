import torch
import torch.nn as nn
import data_load
from pathlib import Path
import numpy as np
from model import OCCI
import torch.nn.functional as F

batch_size = 8
model_path = Path('./trained_occi_300ep.pth')
model = OCCI(batch_size=batch_size, num_slots=3, slot_size=data_load.const_len, Nc=26, Np=4, use_imagine=False)
model.load_state_dict(torch.load(model_path))

eval_loader = data_load.eval_loader
f = open(Path('prediction.txt'), 'w')

im_size = 20
model.eval()

with torch.no_grad():
    hit_num = 0
    tot_num = 0
    for idx, samples in enumerate(eval_loader):
        _, pred_out  = model(samples, idx)
        query_out = samples['query_o'].reshape(batch_size,1,im_size, im_size).numpy()
        # f.write('id {:d}'.format(idx + 1) + '\n')
                
        for i in range(batch_size):
            # process prediction
            pred = F.softmax(pred_out[i], dim=0)
            pred_img = np.argmax(pred.numpy(), axis=0)
            
            query_img = query_out[i][0]
            if (pred_img == query_img).all():
                hit_num += 1
            tot_num += 1
            f.write('{"query_output": \n')
            f.write(np.array2string(query_img, separator=', ',formatter={'float_kind':lambda x: "%d" % x}) + '\n')
            f.write('"pred_output": \n')
            f.write(np.array2string(pred_img, separator=', ',formatter={'float_kind':lambda x: "%d" % x}) + '}\n')
            break
        break
    acc = hit_num / tot_num
    print(f"accuracy: {acc:>3f}")
    f.write(f"accuracy: {acc:>3f}")

f.close()