import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.kernel_size = (5,5)
        self.stride = (2,2)
        self.leakiness = 0.2
        self.dropout_rate = 0.5
        self.encoder_channels = [(1,16), (16,32), (32,64), (64, 128), (128,256), (256,512)]
        self.decoder_channels = [(512, 256), (512, 128), (256, 64), (128,32), (64,16)]
        self.depth = 6
        self.encoders = nn.ModuleDict()
        self.decoders = nn.ModuleDict()
        
        for channel in self.encoder_channels:
            self.encoders.append(self._encoder_bolock(dim_in=channel[0], dim_out=channel[1]))
            
        for i, channel in enumerate(self.decoder_channels):
            drop_out = True if i < 3 else False
            self.decoders.append(self._decoder_block(dim_in=channel[0], dim_out=channel[1], drop_out=drop_out))
            
        self.last_layer = nn.ConvTranspose2d(32, 1, self.kernel_size, self.stride)
            
    def _encoder_bolock(self, dim_in, dim_out):
        return nn.Sequential(
            nn.Conv2d(dim_in, dim_out, self.kernel_size, self.stride, self.padding),
            nn.BatchNorm2d(dim_out),
            nn.LeakyReLU(self.leakiness))
    
    def _decoder_block(self, dim_in, dim_out, drop_out):
        if drop_out:
           return nn.Sequential(
               nn.ConvTranspose2d(),
               nn.BatchNorm2d(),
               nn.Dropout2d(),
               nn.ReLU())
        else:
           return nn.Sequential(
               nn.ConvTranspose2d(), 
               nn.BatchNorm2d(), 
               nn.ReLU())
       
    def _preporcess(self, ):
    def _postprocess(self, ):
            
    def forward(self, input):
        
        outputs = []
        outputs.append(input)
        
        # encoder
        for i in range(self.depth):
            prev_output = outputs[-1]
            outputs.append(self.encoders[i](prev_output))
        
        # decoder
        outputs.append(self.decoders[0](outputs[-1]))
        for i in range(self.depth-2):
            prev_output = outputs[-1]
            outputs.append(self.decoders[i+1](torch.cat([prev_output,outputs[-(i+3)]], dim=1)))
            
        outputs.append(self.last_layer(torch.cat([outputs[-1],outputs[1]])))
        
        return F.sigmoid(self.last_layer(outputs[-1]))
            
        
    