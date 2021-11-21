# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 23:00:18 2021

@author: yipji
"""



import DeepFish.src.models as md


if __name__ == '__main__':
    from torchsummary import summary
    net = md.get_model(model_name = 'resnet', exp_dict = {'dataset': None})
    net.cuda()
    summary(net, (3,224,224))