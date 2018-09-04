# -*- coding: utf-8 -*-
"""
Created on Wed Aug 15 13:57:37 2018

@author: admin
"""

import torch.nn as nn
from ConvLayer import ConvLayer


class Up_Unit(nn.Module):
    def __init__(self,inchannels,outchannels):
        super(Up_Unit,self).__init__()
        
        self.up = nn.Sequential(
                nn.Upsample(scale_factor=2,mode='nearest'),
                ConvLayer(inchannels,outchannels,3,1),
                nn.BatchNorm2d(outchannels,affine=True),
                nn.ReLU(inplace=True)
                )
        
    def forward(self,x):
        x = self.up(x)
        
        return x
