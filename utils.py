import torch
import torch.nn as nn


class L2Loss(nn.Module):
    """Loss function for DeepDream is defined as the L2 norm of a predfined layer"""

    def __init__(self):
        super(L2Loss, self).__init__()

    def forward(self, x):
        return torch.norm(x, p='fro')


class Normalize(nn.Module):
    """Normalize each image with the mean and std of the VGG network

    Images have to be normalized with this mean and std to be compatible
    with the learned parameters of the pretrained network

    Extends:
        nn.Module
    """

    def __init__(self, device):

        super(Normalize, self).__init__()

        # vgg mean and std
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1).float().to(device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1).float().to(device)

    def forward(self, img):

        return (img - self.mean) / self.std


def build_model(pretrained, optim_layer, device):
    """Build the DeepDream model by sequentially adding pretrained model layers

    Parameters
    ----------
        pretrained: torchvisions.models
            pretrained CNN model
        optim_layer: int
            index of the CNN layer to optimize for deepdream
        device: torch.device
            GPU (with number) or CPU
    """

    model = nn.Sequential()
    count = 0

    # normalize image with vgg mean and std
    model.add_module('norm', Normalize(device))

    # rebuild model until 'optim_layer'
    for i, l in enumerate(pretrained.children()):
        model.add_module('{}'.format(i), l)

        # count CNN layers
        if isinstance(l, nn.Conv2d):
            if count == optim_layer:
                print("break")
                break
            count += 1
    return model


def process_tensor(input_img, device):
    """convert numpy array to tensor and send to device

    Parameters
    ----------
        input_img: np.array
            numpy array containing image
        device: torch.device
            GPU (with number) or CPU
    """

    input_img = torch.tensor(input_img).unsqueeze(0)
    input_img = input_img.type(torch.FloatTensor)
    input_img = input_img.to(device)
    input_img.requires_grad = True

    return input_img
