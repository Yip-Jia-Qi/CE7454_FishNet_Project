# -*- coding: utf-8 -*-
"""
Created on Sat Oct 30 19:05:13 2021

@author: yipji
"""

import argparse
from haven import haven_utils as hu
import FishNet.models.net_factory as nf
import pprint
import torch
import torch.nn as nn
import exp_configs
from DeepFish.src import utils as ut
from DeepFish.src import models
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler
from torchvision import transforms


from DeepFish.src.datasets.get_dataset import get_dataset

from DeepFish.src import wrappers

from FishNet150_count import FishNet150_count

import pandas as pd

import time

import torch.optim as optim
import torch.optim.lr_scheduler as lr

from fishy_utils import predict_acc, BCEDiceLoss, CrossEntropyLoss2d, MultiClass_FocalLoss

import matplotlib.pyplot as plt

import torchvision.models
    
def resnet50():
    net = torchvision.models.resnet50(pretrained= True)
    for param in net.parameters():
        param.requires_grad = False
    for param in net.conv1.parameters():
        param.requires_grad = True
    for param in net.bn1.parameters():
        param.requires_grad = True 
    for param in net.layer1.parameters():
        param.requires_grad = True 
    for param in net.layer2.parameters():
        param.requires_grad = True 
    for param in net.layer4.parameters():
        param.requires_grad = True
    net.fc = nn.Sequential(
                        nn.Linear(in_features=2048, out_features=768, bias=True),
                        nn.Linear(in_features=768, out_features=512, bias=True),
                        nn.Linear(in_features=512, out_features=1, bias=True),
                        nn.ReLU(),
                        )    
    return net

def predict_mae(net,loader):
    
    total_mae = 0
    total_batch = 0
    for i in loader:
        # images = i['images'].to(device)
        # labels = i['counts'].to(device)
        # prediction = net(images).round().squeeze()
        mae = torch.mean(abs(i['counts'].to(device) - net(i['images'].to(device)).round().squeeze()))
        total_mae += mae
        total_batch += 1
    
    final_mae = total_mae/total_batch
    
    return final_mae

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--datadir',
                        type=str, default='C:/Users/yipji/Offline Documents/Big Datasets/DeepFish/DeepFish/')
    parser.add_argument("-e", "--exp_config", default='reg')
    parser.add_argument("-uc", "--use_cuda", type=int, default=0)
    args = parser.parse_args()

    device = torch.device('cuda' if args.use_cuda else 'cpu')

    exp_dict = exp_configs.EXP_GROUPS[args.exp_config][0]
    
    # Dataset
    test_set = get_dataset(dataset_name=exp_dict["dataset"], split="test",
                                   transform="resize_normalize",
                                   datadir=args.datadir)

    
    # Load train loader, val loader, and vis loader
   
    test_loader = DataLoader(test_set, shuffle=False, batch_size=exp_dict["batch_size"])


    ##FISHNET MOEL##
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = resnet50()
    net.to(device)
    
    net.load_state_dict(torch.load('./models/Run17_resnet50-regL1_75ep_state.pth'))
        
    with torch.no_grad():
        test = predict_mae(net,test_loader)
        
    print(test)
    
