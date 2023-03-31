import torch
import torch.nn as nn
import data_load
from pathlib import Path
import numpy as np

from model import OCCI
from torch.utils.tensorboard import SummaryWriter

        
writer = SummaryWriter()
train_loader = data_load.data_loader
test_loader = data_load.eval_loader

batch_size = 32
im_size = 20
model = OCCI(num_slots=3, slot_size=data_load.const_len, Nc=26, Np=4, use_imagine=False)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

train_step = 0
test_step = 0

def evaluate(model):
    model.eval()
    global test_step
    
    with torch.no_grad():
        tot_num = 0
        hit_num = 0
        tot_L_test = torch.tensor(0.0) 
        for idx, samples in enumerate(test_loader):
            L_test, pred_out  = model(samples)

            tot_L_test += L_test
            
            test_step += 1
            # writer.add_scalar('loss/test', L_test, test_step)
            # print(f"loss/test: {L_test:>7f}")
            # query_out = samples['query_o'].reshape(batch_size, 1, im_size, im_size).numpy()      
            query_out = samples['query_o'].reshape(-1, 1, im_size, im_size).numpy()      
            batch_size = query_out.shape[0]
            for i in range(batch_size):
                # process prediction
                pred = pred_out[i]
                pred_img = np.argmax(pred.numpy(), axis=0)
                
                query_img = query_out[i][0]
                if (pred_img == query_img).all():
                    hit_num += 1
                tot_num += 1
        acc = hit_num / tot_num
        L_test = tot_L_test / tot_num
        
    model.train()
    return acc, L_test


for epoch in range(300):
    print('Epoch {:d}'.format(epoch + 1))
    # if epoch == 200:
    #   optimizer.lr = 0.0001
        # model.use_imagine = True
    for batch_idx, batched_samples in enumerate(train_loader):
        L_tot, _ = model(batched_samples)
        
        optimizer.zero_grad()
        L_tot.backward()
        optimizer.step()
        train_step += 1
        writer.add_scalar('loss/train', L_tot, train_step)
        # print(f"loss/test: {L_tot:>7f}")
        
        acc, L_test = evaluate(model)
        # writer.add_scalar('acc/test', acc, epoch)
        writer.add_scalar('acc/test', acc, train_step)
        writer.add_scalar('loss/test', L_test, train_step)
        print(f"loss/test: {L_test:>7f}")
    
    ''' add checkpoint '''
    if epoch % 50 == 49:
        model_path = Path('./trained_occi_300ep_{}.pth'.format(int(epoch/50)))
        torch.save(model.state_dict(), model_path)
        
writer.close()    

