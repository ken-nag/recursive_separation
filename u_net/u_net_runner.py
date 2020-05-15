import torch
import sys
import time
sys.path.append('../')
from models.wave_u_net import WaveUNet
from data_utils.slakh_dataset import SlakhDataset
from utils.loss import SiSNRLoss
from utils.wave_net_utils import Utils
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
   
class WaveUNetRunner(Utils):
    def __init__(self, cfg):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dtype= torch.float32

        self.inst_num = cfg['inst_num']
        self.train_data_num = cfg['train_data_num']
        self.valid_data_num = cfg['valid_data_num']
        self.sample_len = cfg['sample_len']
        self.epoch_num = cfg['epoch_num']
        self.train_batch_size = cfg['train_batch_size']
        self.valid_batch_size = cfg['valid_batch_size']
        
        self.train_dataset = SlakhDataset(inst_num=self.inst_num, data_num=self.train_data_num, sample_len=self.sample_len, folder_type='train')
        self.valid_dataset = SlakhDataset(inst_num=self.inst_num, data_num=self.valid_data_num, sample_len=self.sample_len, folder_type='validation')
        
        self.train_data_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.train_batch_size, shuffle=True)
        self.valid_data_loader = torch.utils.data.DataLoader(self.valid_dataset, batch_size=self.valid_batch_size, shuffle=True)
        self.model = WaveUNet(self.sample_len).to(self.device)
        self.criterion = SiSNRLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.save_path = 'results/model/inst_num_{0}/'.format(self.inst_num)
            
    def _run(self, model, criterion, data_loader, batch_size, mode=None):
        running_loss = 0
        for i, (mixtures, sources, _) in enumerate(data_loader):
            mixtures = mixtures.reshape(batch_size, -1).to(self.dtype).to(self.device)
            sources = sources.to(self.dtype).to(self.device)
            model.zero_grad()
            est_sources, est_accompany = model(mixtures)
            
            if mode == 'train' or mode == 'validation':
                true_sources = self.centre_crop(est_sources, sources)
                true_mixtures = self.centre_crop(est_sources, mixtures.unsqueeze(1))
                true_res_sources = true_mixtures.repeat(1,self.inst_num,1) - true_sources
                loss = criterion(est_sources, est_accompany, true_sources, true_res_sources, self.inst_num)
                print(loss.data)
                running_loss += loss.data
                if mode == 'train':
                    loss.backward()
                    self.optimizer.step()
            
        return (running_loss / (i+1))
    
    def train(self):
        train_loss = np.array([])
        valid_loss = np.array([])
        print("start train")
        for epoch in range(self.epoch_num):
            # train
            print('epoch{0}'.format(epoch))
            start = time.time()
            self.model.train()
            tmp_train_loss = self._run(self.model, self.criterion, self.train_data_loader, self.train_batch_size, mode='train')
            train_loss = np.append(train_loss, tmp_train_loss.clone().numpy())
            # validation
            self.model.eval()
            with torch.no_grad():
               tmp_valid_loss = self._run(self.model, self.criterion, self.valid_data_loader, self.valid_batch_size, mode='validation')
               valid_loss = np.append(valid_loss, tmp_valid_loss.clone().numpy())
                 
            if (epoch + 1) % 10 == 0:
                torch.save(self.model.state_dict(), self.save_path + 'wave_u_net{0}.ckpt'.format(epoch + 1))
            
            end = time.time()
            print('----excute time: {0}'.format(end - start))
            plt.plot(train_loss)
            print(train_loss)
            plt.show()
                        
if __name__ == '__main__':
    from configs.train_inst_num2_config import cfg as train_cfg
    obj = WaveUNetRunner(train_cfg)
    obj.train()