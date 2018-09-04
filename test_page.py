# -*- coding: utf-8 -*-
"""
Created on Fri Aug 17 16:34:15 2018

@author: cvpr
"""

from AlignNet import AlignNet
import torch

from torchvision import transforms
from PIL import Image


transform = transforms.Compose([
        transforms.Resize([828,621]),
        transforms.RandomCrop([768,576]),
        transforms.RandomRotation([-5,5]),
        #transforms.RandomHorizontalFlip(0.2),
        transforms.ToTensor()
        #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

img = Image.open('H:\\pedestrian_RGBT\\kaist-rgbt\\images\\set1\\V002\\visible\\I00000.jpg')

out = transform(img)


#device = torch.device("cuda")
#
#res = AlignNet().cuda()
#
#r = torch.randn(4,3,768,576).to(device)
#t = torch.randn(4,3,768,576).to(device)
#
#out = res(r,t)