# -*- coding: utf-8 -*-
"""
Created on Thu Oct 28 00:02:12 2021

@author: yipji
"""

import torch
import models.net_factory as nf
from collections import OrderedDict

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = nf.fishnet150()


checkpoint = torch.load("./checkpoints/fishnet150_ckpt.tar")
# best_prec1 = checkpoint['best_prec1']
state_dict = checkpoint['state_dict']

new_state_dict = OrderedDict()

for k, v in state_dict.items():
    name = k[7:] # remove `module.`
    new_state_dict[name] = v

net.load_state_dict(new_state_dict)


net.to(device)
