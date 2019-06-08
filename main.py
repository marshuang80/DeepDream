import torch
import torch.nn as nn 
import torchvision.models as models
import numpy as np
import matplotlib.pyplot as plt
import tqdm
import argparse
import utils
import logger
from scipy.ndimage.filters import gaussian_filter

def main(args):

    # device
    device = torch.device(args.device)

    # tensorboard
    logger_tb = logger.Logger(log_dir=args.experiment_name)
    
    # load img
    img = plt.imread(args.input_img)
    norm = lambda x: (x - x.min(axis=(0,1))) / (x.max(axis=(0,1)) - x.min(axis=(0,1)))
    img = norm(img)
    img = np.transpose(img, (2,0,1))
    img = torch.tensor(img).unsqueeze(0)
    img = img.type(torch.FloatTensor)
    img = img.to(device)
    img.requires_grad = True 

    # TODO tensorboard

    # cuda 
    device = torch.device(args.device)

    # load pretrained model
    vgg19 = models.vgg19(pretrained=True).features.eval()

    model = utils.build_model(vgg19, optim_layer=15, device=device)
    model = model.to(device)

    loss_fn = utils.L2Loss()

    input_img = img.clone()

    for epoch in range(args.epoch):

        output = model.forward(input_img)
        loss = loss_fn(output)
        loss.backward()
        print('-'*80)
        grad = img.grad.cpu().numpy()
        print(loss)
        print('-'*80)

        sigma = (epoch * 6.0) / args.epoch + 0.5
        grad_smooth1 = gaussian_filter(grad, sigma=sigma)
        grad_smooth2 = gaussian_filter(grad, sigma=sigma*2)
        grad_smooth3 = gaussian_filter(grad, sigma=sigma*0.5)
        grad = (grad_smooth1 + grad_smooth2 + grad_smooth3)
        grad = np.abs(grad).mean() * grad

        grad = torch.Tensor(grad).to(device)

        #input_img += args.lr * grad
        input_img += args.lr * grad

        input_img.data.clamp_(0,1)
        logger_tb.update_loss('loss ', loss.item(), epoch)
        logger_tb.update_image('transformation ', input_img.squeeze().cpu().detach().numpy(), epoch)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_img', type=str, default="./data/clouds.jpeg")
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--experiment_name', type=str, default='test')
    parser.add_argument('--lr', type=float, default=1e-3)

    args = parser.parse_args()

    main(args)

