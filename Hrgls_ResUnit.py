# -*- coding: utf-8 -*-
"""
Created on Wed Aug 15 08:57:41 2018

@author: admin
"""


import torch.nn as nn
from ConvLayer import ConvLayer


class Hrgls_ResUnit(nn.Module):
    def __init__(self, inchannels, outchannels):
        super(Hrgls_ResUnit,self).__init__()
        self.bn1 = nn.BatchNorm2d(inchannels,affine=True)
        self.conv1 = ConvLayer(inchannels,inchannels,1,1)
        
        self.bn2 = nn.BatchNorm2d(inchannels,affine=True)
        self.conv2 = ConvLayer(inchannels,outchannels,3,1)
        
        self.bn3 = nn.BatchNorm2d(outchannels,affine=True)
        self.conv3 = ConvLayer(outchannels,outchannels,1,1)
        
        self.bn4 = nn.BatchNorm2d(inchannels,affine=True)
        self.conv4 = ConvLayer(inchannels,outchannels,1,1)
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self,x):
        left = self.conv4(self.bn4(x))
        
        right = self.conv1(self.relu(self.bn1(x)))
        right = self.conv2(self.relu(self.bn2(right)))
        right = self.conv3(self.relu(self.bn3(right)))
        
        right = right + left
        
        return right
        
        
        
        
        
        
        
        