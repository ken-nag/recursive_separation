import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append('../')
from utils.wave_net_utils import Utils
    
class _DownsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, leakiness):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride) #padding bias?
        self.leaky_relu = nn.LeakyReLU(leakiness)
        
    def _downsampling(self, x):
        return x[:,:,::2]
        
    def forward(self,input):
        h1 = self.leaky_relu(self.conv(input))
        h2 = self._downsampling(h1)
        return h1, h2
        
class _UpsampleBlock(nn.Module, Utils):
    def __init__(self, in_channels, out_channels, kernel_size, stride, leakiness):
        super().__init__()
        self.leaky_relu = nn.LeakyReLU(leakiness)
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride)
        # self.upsample_layer = _LearnUpsampleLayer((batch_size), channels_size)
    
    def _upsampling(self, x):
        channel, batch, source_len = x.shape
        return F.upsample(x, size=(source_len*2 -1), mode='linear')
  
    def forward(self, input, ds_feature):
        us_feature = self._upsampling(input)
        h1 = self.cat_operater(us_feature, ds_feature)
        h2 = self.leaky_relu(self.conv(h1))
        return h2
                                  
# class _LearnUpsampleLayer(nn.Module):
#     def __init__(self, channels_size):
#         super().__init__()
#         self.init_weights = torch.randn((channels_size, source_len, 2), dtype=torch.float32, requires_grad=True)
        
#     def forward(self):
#         weights = torch.sigmoid(self.init_weights)
#         counter_weights = 1.0 - torch.sigmoid(weights)
#         counter_weights.requires_grad = False
        
class WaveUNet(nn.Module, Utils):
    def __init__(self, sample_len=147443):
        super().__init__()
        self.depth = 12
        self.Lm = sample_len
        self.Ls = 16389
        self.ds_kernel_size = 15
        self.us_kernel_size = 5
        self.stride = 1
        self.leakiness = 0.2
        self.growth_rate = 24
        
        self.ds_blocks = nn.ModuleList()
        for i in range(self.depth):
            in_channels = i * self.growth_rate if not i == 0 else 1
            out_channels = (i + 1) * self.growth_rate
            self.ds_blocks.append(_DownsampleBlock(in_channels=in_channels,
                                                   out_channels=out_channels,
                                                   kernel_size=self.ds_kernel_size,
                                                   stride=self.stride,
                                                   leakiness=self.leakiness))
        
        self.bridge_layer = _DownsampleBlock(in_channels=self.depth*self.growth_rate, 
                                             out_channels=(self.depth + 1)*self.growth_rate,  
                                             kernel_size=self.ds_kernel_size,
                                             stride=self.stride,
                                             leakiness=self.leakiness)
        
        self.up_blocks = nn.ModuleList()
        for i in range(self.depth, 0, -1):
            in_channels = ((i + 1) * self.growth_rate) + (i * self.growth_rate)
            out_channels = i * self.growth_rate
            self.up_blocks.append(_UpsampleBlock(in_channels=in_channels, 
                                                 out_channels=out_channels, 
                                                 kernel_size=self.us_kernel_size, 
                                                 stride=self.stride, 
                                                 leakiness=self.leakiness))
        
        
        self.last_layer = nn.Sequential(nn.Conv1d(25, 1, 1, self.stride),
                                        nn.Tanh())
        
    def forward(self, input):
        input = input.unsqueeze(1)
        conv_features = []
        output_features = []
        conv_features.append(input)
        output_features.append(input)
        
        for ds_block in self.ds_blocks:
            conv_feature, output_feature = ds_block(output_features[-1])
            output_features.append(output_feature)
            conv_features.append(conv_feature)
            
        conv_feature, _ = self.bridge_layer(output_features[-1])
        output_features.append(conv_feature)

        for i, up_block in enumerate(self.up_blocks):
            output_feature = up_block(output_features[-1], conv_features[-(i+1)])
            output_features.append(output_feature)
            
        est_source = self.last_layer(self.cat_operater(output_features[-1], conv_features[0]))
        accompany_source = self.centre_crop(est_source, input) - est_source
        
        return est_source, accompany_source
    