import json
import math
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

import IPython

# data_path = Path('data')

# train_path = data_path / 'training'
# eval_path  = data_path / 'evaluation'

train_path = Path('mimg')
eval_path = Path('eval')

train_tasks = {task.stem: json.load(task.open()) for task in train_path.iterdir()}
eval_tasks = {task.stem: json.load(task.open()) for task in eval_path.iterdir()}

const_len = 400
im_size = int(math.sqrt(const_len))
io_number = 5

# padding zeros to im_size * im_size
def padding_zeros(arr, im_size):
    row = arr.shape[0]
    col = arr.shape[1]
    
    if col < im_size:
        for _ in range(im_size - col):
            arr = np.c_[arr, np.zeros(row)]
    else:
        arr = arr[:, :im_size]
        
    if row < im_size:
        for _ in range(im_size - row):
            arr = np.r_[arr, np.zeros(im_size).reshape(1,im_size)]
    else: 
        arr = arr[:im_size, :]
    
    return arr


def preprocess(origin_tasks, im_size):
    modified_tasks = []
    for task in origin_tasks.values():
        # record all inputs(resp.output) of a task
        # shape of [io_num, input_size](resp.[io_num, output_size])
        input  = torch.tensor([])
        output = torch.tensor([])
        
        io_num = 0
        for item in task['train']:
            if io_num == io_number:
                break
            
            inp = padding_zeros(np.asarray(item['input']), im_size)
            out = padding_zeros(np.asarray(item['output']), im_size)
                
            inp = inp.flatten()
            out = out.flatten()
            
            inp = torch.DoubleTensor(inp)[None,:]
            out = torch.DoubleTensor(out)[None,:]
            
            input = torch.cat((input, inp), dim=0)
            output = torch.cat((output, out), dim=0)
            io_num += 1
        
        if io_num < 5:
            for _ in range(io_number - io_num):
                inp = torch.zeros(const_len, dtype=torch.float64)[None,:]
                out = torch.zeros(const_len, dtype=torch.float64)[None,:]
                input = torch.cat((input, inp), dim=0)
                output = torch.cat((output, out), dim=0)
        
        # process query io
        query_i = padding_zeros(np.asarray(task['test'][0]['input']), im_size)
        query_o = padding_zeros(np.asarray(task['test'][0]['output']), im_size)
        query_i = query_i.flatten()
        query_o = query_o.flatten()
        
        query_i = torch.DoubleTensor(query_i)[None,:]
        query_o = torch.DoubleTensor(query_o)[None,:]
        
        # modified_tasks is a list of dict, each dict represents a task
        modified_tasks.append({'input': input, 'output': output, 'query_i': query_i, 'query_o': query_o})
            
    return modified_tasks    


class ArcDataset(Dataset):
    def __init__(self, train_io_set):
        self.train_io_set = train_io_set
        # self.transform = transform
        
    def __len__(self):
        return len(self.train_io_set)
        
    def __getitem__(self, idx):
        sample = self.train_io_set[idx]
        return sample

train_dataset = ArcDataset(preprocess(train_tasks, im_size))
data_loader = DataLoader(train_dataset, batch_size = 32, shuffle = True, drop_last = False)

eval_dataset = ArcDataset(preprocess(train_tasks, im_size))
eval_loader = DataLoader(eval_dataset, batch_size = 32, shuffle = False, drop_last = False)
