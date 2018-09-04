# -*- coding: utf-8 -*-
"""
Created on Wed Aug 15 15:04:39 2018

@author: admin
"""

import torchvision
import torch.nn as nn



class Res_Module(nn.Module):
    def __init__(self):
        super(Res_Module,self).__init__()
        
        self.res = torchvision.models.resnet34(pretrained=True)
        
    def forward(self,x):
        
        for name,module in self.res._modules.items():
            if name == 'layer4':
                break
            x = module(x)
        
        return x