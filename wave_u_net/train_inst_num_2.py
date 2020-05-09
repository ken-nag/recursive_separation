import torch
import sys
sys.path.append('../')
from models.wave_u_net import WaveUNet
from data_utils.slakh_dataset import SlakhDataset
from utils.loss import SiSNRLoss
from tqdm import tqdm

inst_num = 2
train_data_num = 4
valid_data_num = 10
epoch_num = 1000
train_batch_size = 2
valid_batch_size = 5
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dtype= torch.float32
train_dataset = SlakhDataset(inst_num=inst_num, data_num=train_data_num, folder_type='train')
valid_dataset = SlakhDataset(inst_num=inst_num, data_num=valid_data_num, folder_type='validation')

train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
valid_data_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=valid_batch_size, shuffle=True)
model = WaveUNet().to(device)
criterion = SiSNRLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
# to_do scheduler

for epoch in tqdm(range(epoch_num)):
    for i, (mixtures, sources, _) in enumerate(train_data_loader):
        mixtures = mixtures.reshape(train_batch_size, -1).to(dtype).to(device)
        sources = sources.reshape(train_batch_size, -1).to(dtype).to(device)
        
        est_sources, accompany = model(mixtures)
        loss = criterion(est_sources, accompany, sources, inst_num)
        model.zero_grad()
        loss.backward()
        optimizer.step()
    

    #to_do visualize
   
    
    if (epoch + 1) / 10 == 0:
        torch.save(model.state_dict(), 'results/model/wave_u_net{0}.ckpt'.format(epoch))