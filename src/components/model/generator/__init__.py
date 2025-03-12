import torch.nn as nn
import torch
from src.exception.exception import ExceptionNetwork, sys

class Generator(nn.Module):
    
    def __init__(self,channel_size_in,channel_size_out):
        super(Generator,self).__init__()
        
        self.e1=nn.Sequential(nn.Conv2d(channel_size_in,64,4,2,padding=1,padding_mode="reflect"),
                              nn.LeakyReLU(0.2))
        
        self.e2=self.Encoder(64,128)
        self.e3=self.Encoder(128,256)
        self.e4=self.Encoder(256,512)
        self.e5=self.Encoder(512,512)
        self.e6=self.Encoder(512,512)
        self.e7=self.Encoder(512,512)
        self.e8=self.Encoder(512,512)
        
        self.d1=self.Decoder(512,512,True)
        self.d2=self.Decoder((512+512),1024,True)
        self.d3=self.Decoder((1024+512),1024,True)
        self.d4=self.Decoder((1024+512),1024,False)
        self.d5=self.Decoder((1024+512),1024,False)
        self.d6=self.Decoder((1024+256),512,False)
        self.d7=self.Decoder((512+128),256,False)
        self.d8=self.Decoder((256+64),128,False)
        
        self.out_layer=nn.Sequential(nn.Conv2d(128,channel_size_out,1),
                                     nn.Tanh())

        
    
    def Encoder(self,in_channels,out_channels):
        return nn.Sequential(nn.Conv2d(in_channels,out_channels,4,2,padding=1,padding_mode="reflect"),
                              nn.BatchNorm2d(out_channels),
                              nn.LeakyReLU(0.2))
        
    def Decoder(self,in_channels,out_channels,dropout:bool):
        if dropout==False:
            return nn.Sequential(nn.ConvTranspose2d(in_channels,out_channels,4,2,padding=1),
                              nn.BatchNorm2d(out_channels),
                              nn.ReLU())
            
        if dropout==True:
            return nn.Sequential(nn.ConvTranspose2d(in_channels,out_channels,4,2,padding=1),
                              nn.BatchNorm2d(out_channels),
                              nn.Dropout(0.5),
                              nn.ReLU())
        
    
    def forward(self,data):
        try: 
            x1=self.e1(data)
            x2=self.e2(x1)
            x3=self.e3(x2)
            x4=self.e4(x3)
            x5=self.e5(x4)
            x6=self.e6(x5)
            x7=self.e7(x6)
            x8=self.e8(x7)
            
            db1=self.d1(x8)
            
            db2=self.d2(torch.cat([db1,x7],dim=1))
            db3=self.d3(torch.cat([db2,x6],dim=1))
            db4=self.d4(torch.cat([db3,x5],dim=1))
            db5=self.d5(torch.cat([db4,x4],dim=1))
            db6=self.d6(torch.cat([db5,x3],dim=1))
            db7=self.d7(torch.cat([db6,x2],dim=1))
            db8=self.d8(torch.cat([db7,x1],dim=1))
            
            return self.out_layer(db8)
        
        except Exception as e:
            raise ExceptionNetwork(e,sys)