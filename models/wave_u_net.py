import torch
import torch.nn as nn

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
        
class _UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, leakiness, last_layer=False):
        super().__init__()
        self.tanh = nn.Tanh()
        self.leaky_relu = nn.LeakyReLU(leakiness)
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride)
        self.last_layer = last_layer
        # self.upsample_layer = _LearnUpsampleLayer((batch_size), channels_size)
    
    def _upsampling(self, x):
        pass
    
    def forward(self, input):
        if not self.last_layer:
            h1 = self.leaky_relu(self.conv(input))
        else:
            h1 = self.tanh(self.conv(input))
        
        h2 = self._upsampling(h1)
        return h2
        
                            
class _LearnUpsampleLayer(nn.Module):
    def __init__(self, channels_size):
        super().__init__()
        batch_size, channels_size, 
        self.init_weights = torch.randn((channels_size, 2), dtype=torch.float32, requires_grad=True)
        
    def forward(self):
        counter_weights = 1.0 - torch.sigmoid(self.init_weights)
        
        
        
class WaveUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.depth = 12
        self.Lm = 147443
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
        
        
        self.last_layer = _UpsampleBlock(in_channels=25,
                                         out_channels=1,
                                         kernel_size=self.us_kernel_size,
                                         stride=self.stride,
                                         leakiness=self.leakiness,
                                         last_layer=True)
        

    def _centre_crop(self, us_feature, ds_feature):
        d = ds_feature.shape[2] - us_feature.shape[2]
        return ds_feature[:,:,d//2:-d+(d//2)]
        
    
    def _cat_operater(self, us_feature, ds_feature):
        print('us_feature:',us_feature)
        print('ds_feature:', ds_feature)
        cropped_feature = self._centre_crop(us_feature, ds_feature)
        return torch.cat([cropped_feature, us_feature], dim=1)
        
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
            
        _, output_feature = self.bridge_layer(output_features[-1])
        output_features.append(output_feature)
        
        i = 0
        for up_block in self.up_blocks:
            input_feature = self._cat_operater(output_features[-1], conv_features[-1])
            output_feature = up_block(input_feature)
            output_features.append(output_feature)
            i = i + 1
            print("iter:", i)
            
        last_input_feature = self._cat_operater(output_features[-1], output_features[0])
        est_source = self.last_layer(last_input_feature)
        accompany_source = self._centre_crop(est_source, input) - est_source
        
        return est_source, accompany_source
    