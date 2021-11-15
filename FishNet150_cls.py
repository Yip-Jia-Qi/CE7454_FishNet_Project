# -*- coding: utf-8 -*-
"""
Created on Tue Nov  2 16:18:42 2021

@author: yipji
"""
import FishNet.models.net_factory as nf
import torch
import torch.nn as nn


class FishNet150_cls(nn.Module):
    def __init__(self, n_classes=1, pretrained = True):
        super().__init__()
        self.n_classes = n_classes
        self.net = nf.fishnet150()
        
        if pretrained:
            checkpoint = torch.load("./FishNet/checkpoints/fishnet150_ckpt.tar")
            # best_prec1 = checkpoint['best_prec1']
            state_dict = checkpoint['state_dict']
        
            from collections import OrderedDict
            
            new_state_dict = OrderedDict()
            
            for k, v in state_dict.items():
                name = k[7:] # remove `module.`
                new_state_dict[name] = v
            
            self.net.load_state_dict(new_state_dict)
        
        self.net.fish.fish[9][4][1] = nn.Sequential(nn.Flatten(), 
                                                    nn.Linear(in_features=1056, out_features= self.n_classes, bias=True),
                                                    nn.Softmax()
                                                    )
    def forward(self, x):
        return self.net(x)
    
    
if __name__ == '__main__':
    from torchsummary import summary
    net = FishNet150_cls().cuda()
    summary(net, (3,224,224))