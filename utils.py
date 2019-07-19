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
    layer_count = 0 
    flag = False                 
    model.add_module('norm', Normalize(device))
    '''
    for name, child in pretrained.named_children():
        for l in child.children():
            model.add_module('{}'.format(layer_count), l) 
            layer_count += 1
            if isinstance(l, nn.Conv2d):     
                if count == optim_layer:     
                    print("break")
                    flag = True
                    break                    
                count +=1 
        if flag:
            break
    '''
    for i, l in enumerate(pretrained.children()):
        model.add_module('{}'.format(i), l) 
        if isinstance(l, nn.Conv2d):     
            if count == optim_layer:     
                print("break")
                break                    
            count +=1                        
    print(model)
    return model                         

def process_tensor(input_img, device):

    input_img = torch.tensor(input_img).unsqueeze(0)
    input_img = input_img.type(torch.FloatTensor)   
    input_img = input_img.to(device)                
    input_img.requires_grad = True                  
    
    return input_img

