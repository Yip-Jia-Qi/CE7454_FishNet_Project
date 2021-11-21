# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 19:42:50 2021

@author: yipji
"""



import torch
import torch.nn as nn
import torchvision
import numpy as np
import torchvision.transforms as transforms
from torchsummary import summary

from pathlib import Path
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

#homebrew modules
import models.net_factory as nf
from CE7454_Fashion_Dataset_v4 import FashionDataset
from fashion_utils import single_acc, multi_acc, imshow, predict_acc, predict

#%%
#Dataset
datadir = r'C:\Users\yipji\Offline Documents\Big Datasets'
projdir = datadir+r'\CE7454 Fashion Dataset\FashionDataset'
   
img_dir = Path(projdir+r'\img')

x_train = Path(projdir+r'\split\train.txt')
y_train = Path(projdir+r'\split\train_attr.txt')

x_val = Path(projdir+r'\split\val.txt')
y_val = Path(projdir+r'\split\val_attr.txt')
    
x_test = Path(projdir+r'\split\test.txt')

torch.manual_seed(0)

tsfms = transforms.Compose([
    transforms.RandomResizedCrop(224),
    # transforms.CenterCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.RandomErasing(p=0.2),
    # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize((0, 0, 0), (1, 1, 1)),
    ])

tsfms0 = transforms.Compose([
    transforms.CenterCrop(224),
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize((0, 0, 0), (1, 1, 1)),    
    ])


   
train_dataset = FashionDataset(img_dir, x_train, y_train, one_hot = False, transform=tsfms)
# val_dataset = FashionDataset(img_dir, x_val, y_val, one_hot = False, transform=tsfms)
# test_dataset = FashionDataset(img_dir, x_test, y_val, one_hot = False, transform=tsfms0) #using y_val as placeholder since y_test is not available
# val0_dataset = FashionDataset(img_dir, x_val, y_val, one_hot = False, transform=tsfms0)

train_dataloader = DataLoader(train_dataset,batch_size=1,shuffle=True,num_workers=0)
# val_dataloader = DataLoader(val_dataset,batch_size=2,shuffle=True)
# test_dataloader = DataLoader(test_dataset,batch_size=2,shuffle=False)
# val0_dataloader = DataLoader(val0_dataset,batch_size=2,shuffle=True)

# get some random training images
dataiter = iter(train_dataloader)
images, labels = dataiter.next()

# show images
imshow(torchvision.utils.make_grid(images[0:5]))

#%%
#Loading the Model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = nf.fishnet150()


checkpoint = torch.load("./checkpoints/fishnet150_ckpt.tar")
# best_prec1 = checkpoint['best_prec1']
state_dict = checkpoint['state_dict']

from collections import OrderedDict

new_state_dict = OrderedDict()

for k, v in state_dict.items():
    name = k[7:] # remove `module.`
    new_state_dict[name] = v

net.load_state_dict(new_state_dict)

net.fish.fish[9][4][1] = nn.Sequential(nn.Linear(in_features=1056, out_features=42, bias=True),
                                       nn.Unflatten(dim = 1, unflattened_size=(6,7,1)),
                                       nn.Softmax(dim=2),
                                       )    

net.to(device)

#%%

import torchvision
import torch.optim as optim
import torch.optim.lr_scheduler as lr
import time

#RUN BABY RUN

criterion = nn.MSELoss().to(device) #adam learning rate about 0.001 is good
# criterion = FocalLoss(alpha = 10, gamma = 5).to(device) #adam learning rate about 0.000001 is good
optimizer = optim.Adam(net.parameters(), lr=0.0005)
# scheduler1 = lr.ExponentialLR(optimizer, gamma=0.9, verbose = True)
scheduler2 = lr.MultiStepLR(optimizer,milestones=[10,15,20],gamma=0.1, verbose = True)


run_number = 1
model_name = "FishNet150"
total_epochs = 1

#setting up counters#
train_avg_acc_by_epoch = []
val_avg_acc_by_epoch = []
tic0 = time.perf_counter()

for epoch in range(total_epochs):  # loop over the dataset multiple times
    
    ###Epoch start counter
    tic = time.perf_counter()
    
    running_loss = 0.0
    for i, data in enumerate(train_dataloader, 0):
        
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        
        for i in range(len(labels)):
            labels[i] = labels[i].flatten()
        
        inputs, labels = inputs.to(device), labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        outputs = outputs.squeeze()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 20 == 19:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 20))
            running_loss = 0.0
    
    # scheduler1.step()
    scheduler2.step()
    
    ###Validation start counter
    tic2 = time.perf_counter()
    
    # ep_train_acc = predict_acc(net,train_dataloader)
    # ep_val_acc = predict_acc(net,val_dataloader)
    # train_avg_acc_by_epoch.append(ep_train_acc)    
    # val_avg_acc_by_epoch.append(ep_val_acc)
    
    #Epoch End Counter
    toc = time.perf_counter()
    # print(f'train acc: {ep_train_acc:0.4f}')
    # print(f'val acc: {ep_val_acc:0.4f}')
    print(f"Epoch {epoch+1} of {total_epochs} Ended. Total epoch took {toc-tic:0.4f}s. Validation took {toc-tic2:04f}s")

# final_val = predict_acc(net.to(device), val0_dataloader)
# # torch.save(net.state_dict(), f'./Run{run_number}_{model_name}_{total_epochs:0.0f}ep_state.pth')
# plt.plot(train_avg_acc_by_epoch)
# plt.plot(val_avg_acc_by_epoch)
# print(f'Finished Training. Total time {toc-tic0:0.4f}s, Final Validation Acc {final_val:0.4f}')