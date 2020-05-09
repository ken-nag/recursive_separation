import torch
import numpy as np
import sys
sys.path.append('../')
import glob

class SlakhDataset(torch.utils.data.Dataset):
    def __init__(self, inst_num, data_num, transform=None, folder_type=None):
        self.data_num = data_num
        self.transform = transform
        self.npzs_path = glob.glob('../data/slakh_inst{0}/{1}/*'.format(inst_num, folder_type))
       
    def __len__(self):
        return self.data_num
        
    def __getitem__(self, idx):         
        npz_obj = np.load(self.npzs_path[idx])
        mixture = npz_obj['mixture']
        sources = npz_obj['sources']
        inst_num = npz_obj['instruments_num']
        
        if self.transform:
            pass
        
        return mixture, sources, inst_num
        