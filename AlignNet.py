# -*- coding: utf-8 -*-
"""
Created on Wed Aug 15 15:55:27 2018

@author: admin
"""

import torch.nn as nn
from Res_Module import Res_Module
from Matching_Module import Matching_Module
from ResUnit import ResUnit
from Up_Unit import Up_Unit
from ConvLayer import ConvLayer


class AlignNet(nn.Module):
    def __init__(self):
        super(AlignNet,self).__init__()
        
        self.down =Res_Module()
        
        self.match1 = Matching_Module(inchannels = 256)
        
        self.mid1 = nn.Sequential(
                ResUnit(channels=256,kernel=3),
                ResUnit(channels=256,kernel=3)
                )
        
        self.mid2 = nn.Sequential(
                ResUnit(channels=256,kernel=3),
                ResUnit(channels=256,kernel=3)
                )
        
        self.match2 = Matching_Module(inchannels = 256)
        
        self.up1 = Up_Unit(inchannels=256,outchannels=128)
        self.up2 = Up_Unit(inchannels=128,outchannels=64)
        self.up3 = Up_Unit(inchannels=64,outchannels=16)
        self.up4 = Up_Unit(inchannels=16,outchannels=3)
        
        self.final = nn.Sequential(
                ConvLayer(in_channels=3,out_channels=3,kernel_size=7),
                ConvLayer(in_channels=3,out_channels=3,kernel_size=3),
                ConvLayer(in_channels=3,out_channels=3,kernel_size=1)
                )
    
    def forward(self,R,T):
        R_feat = self.down(R)
        T_feat = self.down(T)
        
        R_feat,T_masked = self.match1(R_feat,T_feat)
        
        R_feat = self.mid1(R_feat)
        T_masked = self.mid2(T_masked)
        
        R_feat,T_masked = self.match2(R_feat,T_feat)
        
        T = self.up4(self.up3(self.up2(self.up1(T_masked))))
        T = self.final(T)
        
        return T
        
        
        