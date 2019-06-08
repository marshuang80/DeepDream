import matplotlib.pyplot as plt 
import scipy.misc as misc
import numpy as np
import torch 
import torch.nn as nn

class L2Loss(nn.Module):

    def __init__(self):
        super(L2Loss, self).__init__()

    def forward(self, x):
        return torch.norm(x, p='fro')


class Normalize(nn.Module):

    def __init__(self, device):

         super(Normalize, self).__init__()

         # vgg mean and std
         self.mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1).float().to(device)
         self.std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1).float().to(device)

    def forward(self, img):

        return (img - self.mean) / self.std

def build_model(pretrained, optim_layer, device):
                                         
    model = nn.Sequential()              
    count = 0                            
                                         
    model.add_module('norm', Normalize(device))
                                         
    for i, l in enumerate(pretrained.children()):
        model.add_module('{}'.format(i), l) 
                                         
        if isinstance(l, nn.Conv2d):     
            if count == optim_layer:     
                break                    
        count +=1                        
                                        
    return model                         

