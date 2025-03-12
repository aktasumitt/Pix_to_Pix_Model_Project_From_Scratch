import torch.nn as nn
import torch
from src.exception.exception import ExceptionNetwork, sys

class Discriminator(nn.Module):
    def __init__(self,channel_size_out,channel_size_in):
        super(Discriminator,self).__init__()
        
        self.first_layer=nn.Sequential(nn.Conv2d((channel_size_in+channel_size_out),64,4,2,1,padding_mode="reflect"),
                                       nn.LeakyReLU(0.2))
        
        self.down1=self.down_blocks(64,128)
        self.down2=self.down_blocks(128,256)
        self.down3=self.down_blocks(256,512)
        self.down4=self.down_blocks(512,512)
        self.down5=self.down_blocks(512,512)
        
        self.out_layer=nn.Conv2d(512,1,4,2,1,padding_mode="reflect")
        
    def down_blocks(self,in_channels,out_channels):
        return nn.Sequential(nn.Conv2d(in_channels,out_channels,2,1,padding_mode="reflect"),
                             nn.BatchNorm2d(out_channels),
                             nn.LeakyReLU(0.2))
    def forward(self,x,y):
        try:    
            x=self.first_layer(torch.cat([x,y],dim=1))
            
            x=self.down1(x)
            x=self.down2(x)
            x=self.down3(x)
            x=self.down4(x)
            x=self.down5(x)
            
            out=self.out_layer(x)
            
            return out
        
        except Exception as e:
            raise ExceptionNetwork(e,sys)
