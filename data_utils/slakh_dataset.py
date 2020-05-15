import torch
import numpy as np
import sys
sys.path.append('../')
import glob
import random
random.seed(0)

class SlakhDataset(torch.utils.data.Dataset):
    def __init__(self, inst_num, data_num, sample_len=None, transform=None, folder_type=None):
        self.data_num = data_num
        self.transform = transform
        self.npzs_path = glob.glob('../data/slakh_inst{0}/{1}/*'.format(inst_num, folder_type))
        self.sample_len = sample_len
       
    def __len__(self):
        return self.data_num
        
    def __getitem__(self, idx):
        path = random.sample(self.npzs_path, 1)
        npz_obj = np.load(path[0])
        mixture = npz_obj['mixture']
        sources = npz_obj['sources']
        inst_num = npz_obj['instruments_num']
        
        if self.transform:
            pass
        
        if self.sample_len:
            return mixture[:self.sample_len], sources[:, :self.sample_len], inst_num
        else:
            return mixture, sources, inst_num