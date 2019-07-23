import torch
from torch.nn.modules.upsampling import Upsample
import torchvision.models as models
import numpy as np
import matplotlib.pyplot as plt
import argparse
import utils
import logger
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage import zoom
from skimage.transform import rescale


def main(args):

    # device
    device = torch.device(args.device)

    # tensorboard
    logger_tb = logger.Logger(log_dir=args.experiment_name)

    # load img
    img = plt.imread(args.input_img)

    # def norm(x): return (x - x.min(axis=(0, 1))) / (x.max(axis=(0, 1)) - x.min(axis=(0, 1)))
    norm = lambda x: (x - x.min(axis=(0, 1))) / (x.max(axis=(0, 1)) - x.min(axis=(0, 1)))
    img = norm(img)
    img = np.transpose(img, (2, 0, 1))

    # load pretrained model
    vgg19 = models.vgg19(pretrained=True).features.eval()
    model = utils.build_model(vgg19, optim_layer=args.layer, device=device)
    model = model.to(device)

    # loss function
    loss_fn = utils.L2Loss()

    # Populate oct_imgs with different sized zooms of the original image
    oct_imgs = [img]
    for oct_itr in range(args.num_octave):

        zoom_img = zoom(oct_imgs[-1], (1, 1 / args.octave_ratio, 1 / args.octave_ratio))
        oct_imgs.append(zoom_img)

    oct_imgs = [utils.process_tensor(oct_img, device) for oct_img in oct_imgs]
    ori_oct_imgs = [oct_img.clone() for oct_img in oct_imgs]

    while len(oct_imgs) > 0:
        oct_img = oct_imgs.pop()
        ori_oct_img = ori_oct_imgs.pop()
        idx = len(oct_imgs)

        print(f"Deep dreaming on octave: {idx}")

        for epoch in range(args.epoch):
            model.zero_grad()
            output = model.forward(oct_img)
            loss = loss_fn(output)
            loss.backward()
            grad = oct_img.grad.cpu().numpy()
            lr = args.lr / np.abs(grad).mean()

            # apply gaussian smoothing on gradient
            sigma = (epoch * 4.0) / args.epoch + 0.5
            grad_smooth1 = gaussian_filter(grad, sigma=sigma)
            grad_smooth2 = gaussian_filter(grad, sigma=sigma * 2)
            grad_smooth3 = gaussian_filter(grad, sigma=sigma * 0.5)
            grad = (grad_smooth1 + grad_smooth2 + grad_smooth3)
            grad = torch.Tensor(grad).to(device)

            # backpropagate on ocatve image
            oct_img.data += lr * grad.data
            oct_img.data.clamp_(0, 1)
            oct_img.grad.data.zero_()

            # display image on tensorboard
            dream_img = oct_img.squeeze().cpu().detach().numpy().copy()
            logger_tb.update_loss('loss ', loss.item(), epoch)
            logger_tb.update_image(f'transformation oct{idx}', dream_img, epoch)

        if len(oct_imgs) == 0:
            break

        # add the "dreamed" portion of the current octave to the next octave
        h = oct_imgs[-1].shape[2]
        w = oct_imgs[-1].shape[3]
        difference = oct_img.data - ori_oct_img.data
        difference = Upsample(size=(h, w), mode='nearest')(difference)
        oct_imgs[-1].data += difference


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--input_img', type=str, default="./data/clouds.jpeg")
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--experiment_name', type=str, default='test')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num_octave', type=int, default=2)
    parser.add_argument('--octave_ratio', type=float, default=0.5)
    parser.add_argument('--layer', type=int, default=12)
    parser.add_argument('--model', type=str, default='vgg')

    args = parser.parse_args()

    main(args)
