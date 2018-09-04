# -*- coding: utf-8 -*-
"""
Created on Tue Aug 14 21:28:37 2018

@author: cvpr
"""


import torch.nn as nn
from Hrgls_ResUnit import Hrgls_ResUnit
from ResUnit import ResUnit
from ConvLayer import ConvLayer


class Hourglass(nn.Module):
    def __init__(self, dense_rate, inchannels):
        super(Hourglass,self).__init__()
        
#        self.modules1 = []
#        self.modules2 = []
#        self.modules3 = []
#        self.modules4 = []
        
        
        self.channels = inchannels
        assert dense_rate%2==0,'The dense_rate must be even'
        half_rate = dense_rate//2
        self.rate = half_rate
        
        
#        for i in range(half_rate):
        self.block1 = nn.Sequential(
                nn.MaxPool2d(kernel_size=2, stride=2),
                ResUnit(self.channels,3),
                Hrgls_ResUnit(self.channels, (self.channels*2))
                )
        self.channels = self.channels*2
#            self.modules1.append(block)
        
#        for i in range(half_rate):
        self.block2 = nn.Sequential(
                nn.MaxPool2d(kernel_size=2, stride=2),
                ResUnit(self.channels,3),
                Hrgls_ResUnit(self.channels, (self.channels*2))
                   )
        self.channels = self.channels*2
#            self.modules2.append(block)
            
            
            
            
#        for i in range(half_rate):
        self.block3 = nn.Sequential(
                   ResUnit(self.channels,3),
                   Hrgls_ResUnit(self.channels, (self.channels//2)),
                   
                   nn.Upsample(scale_factor=2,mode='nearest'),
                   ConvLayer((self.channels//2),(self.channels//2),3,1),
                   nn.BatchNorm2d((self.channels//2))
                   )
        self.channels = self.channels//2
#            self.modules3.append(block)
            
#        for i in range(half_rate):
        self.block4 = nn.Sequential(
                   ResUnit(self.channels,3),
                   Hrgls_ResUnit(self.channels, (self.channels//2)),
                   
                   nn.Upsample(scale_factor=2,mode='nearest'),
                   ConvLayer((self.channels//2),(self.channels//2),3,1),
                   nn.BatchNorm2d((self.channels//2))
                   )
        self.channels = self.channels//2
#            self.modules4.append(block)
        
        
        
        self.final = nn.Sequential(
                ResUnit(self.channels,1),
                ResUnit(self.channels,1)
                )
        
        
        
        
    def forward(self, x):
        phase1 = x
#        for i in range(self.rate):
        phase1 = self.block1(phase1)
        
        phase2 = phase1
#        for i in range(self.rate):
        phase2 = self.block2(phase2)
        
        phase3 = phase2
#        for i in range(self.rate):
        phase3 = self.block3(phase3)
            
        phase4 = phase3 + phase1 #residual learning within the phases
#        for i in range(self.rate):
        phase4 = self.block4(phase4)
        
        out = self.final(phase4)
        
        return out
        
        
        
        
        
        