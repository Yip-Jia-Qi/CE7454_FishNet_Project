# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 00:49:06 2021

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

from FishNet150_cls import FishNet150_cls

import pandas as pd

import time

import torch.optim as optim
import torch.optim.lr_scheduler as lr

from fishy_utils import predict_acc, BCEDiceLoss, CrossEntropyLoss2d, MultiClass_FocalLoss

import matplotlib.pyplot as plt



def predict_acc(net,loader):
    
    total_acc = 0
    total_batch = 0
    for i in loader:                
        total_acc += sum((net(i['images'].to(device)).squeeze()>0.5) == i['labels'].to(device))
        total_batch += len(i['labels'])
    
    final_acc = total_acc/total_batch
    
    return final_acc

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--datadir',
                        type=str, default='C:/Users/yipji/Offline Documents/Big Datasets/DeepFish/DeepFish/')
    parser.add_argument("-e", "--exp_config", default='clf')
    parser.add_argument("-uc", "--use_cuda", type=int, default=0)
    args = parser.parse_args()

    device = torch.device('cuda' if args.use_cuda else 'cpu')

    exp_dict = exp_configs.EXP_GROUPS[args.exp_config][0]
    
    # Dataset
    # Load val set and train set

    test_set = get_dataset(dataset_name=exp_dict["dataset"], split="test",
                                   transform="resize_normalize",
                                   datadir=args.datadir)

    
    # Load train loader, val loader, and vis loader

    test_loader = DataLoader(test_set, shuffle=False, batch_size=exp_dict["batch_size"])



    ##FISHNET MOEL##
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = FishNet150_cls().to(device)
    net.load_state_dict(torch.load('./models/Run6_fishnet150-clf_25ep_state.pth'))
    
    with torch.no_grad():
        test = predict_acc(net, test_loader)
    
    print(test)