# -*- coding: utf-8 -*-
"""
Created on Wed Aug 15 12:21:35 2018

@author: admin
"""

import torch.nn as nn
from Hourglass import Hourglass
from ResUnit import ResUnit
from kron import KronEmbed


class Matching_Module(nn.Module):
    def __init__(self,inchannels):
        super(Matching_Module,self).__init__()
        
        self.Hrgls = Hourglass(dense_rate=2, inchannels=inchannels)
        
        self.tunnel = nn.Sequential(
                ResUnit(inchannels,3),
                ResUnit(inchannels,3),
                ResUnit(inchannels,3),
                ResUnit(inchannels,3)
                )
        
        self.kron = KronEmbed()
        
    def forward(self,R_feat,T_feat):
        
        R_feat = self.Hrgls(R_feat)
        
        T_feat = self.tunnel(T_feat)
        
        T_masked = self.kron(R_feat,T_feat)
        
        T_masked = T_masked + T_feat   #residual learning for T_features
        
        return R_feat,T_masked
        
        
        
        