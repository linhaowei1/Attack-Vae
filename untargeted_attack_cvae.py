from utils import get_args
from VAE import VAE
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.utils as vutils
import math
from models.utils import loss_functions as lf
import os
from loguru import logger
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
import pandas as pd
import pdb
from datasets import get_dataset
from models.vae import ConditionalVAE2
from tensorboardX import SummaryWriter
from torch.distributions.multivariate_normal import MultivariateNormal
import random
DATA_PATH = '~/data/'

def untargeted_attack(model, dataloader, device, epsilon=0.5, iters=100, alpha = 1/200):
    model.eval()
    target_img = None
    target_img_label = None
    target_latent = None
    labels_cnt = [0] * 6

    for idx, (img, tgt) in tqdm(enumerate(dataloader), total=len(dataloader), desc="training"):
        img_flatten = img.to(device)
        origin_img = img_flatten.data
        label = int(tgt.item())
        label_cnt = labels_cnt[label]
        y = tgt.long()
        y_onehot = torch.Tensor(y.shape[0], model.class_num)
        y_onehot.zero_()
        y_onehot.scatter_(1,y.view(-1,1), 1)
        y_onehot = y_onehot.to(device)

        for i in range(iters):
            img_flatten.requires_grad = True
            model.zero_grad()

            mu, logvar, recon = model(img_flatten, y_onehot)
            variatL = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
            variatL = lf.weighted_average(variatL, weights=None, dim=0)
            variatL /= (1 * 28 * 28)
            data_resize = img_flatten.reshape(-1, 1 * 28 * 28)
            recon_resize = recon.reshape(-1, 1 * 28 * 28)
            reconL = (data_resize - recon_resize) ** 2
            reconL = torch.mean(reconL, 1)
            reconL = lf.weighted_average(reconL, weights=None, dim=0)
            loss = variatL + reconL
            loss.backward()
            adv_images = img_flatten + alpha*img_flatten.grad.sign()
            eta = torch.clamp(adv_images - origin_img, min=-epsilon, max=epsilon)
            img_flatten = torch.clamp(origin_img + eta, min=0, max=1).detach_()

        _, _, attack_img = model(img_flatten.to(device), y_onehot)
        _, _, recons_img = model(img.to(device), y_onehot)
        output_dir = f'untargeted_attack_cvae_partition1/{label}'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        vutils.save_image(torch.cat([img.cpu(), img_flatten.cpu(), recons_img.cpu(), attack_img.cpu()],dim=0),f'{output_dir}/{label_cnt}.png')
        labels_cnt[label] += 1

def main():
    args = get_args()

    log_file_name = args.log_file
    logger.add(log_file_name)
    logger.info('args:\n' + args.__repr__())

    batch_size = args.batch_size
    output_dir = args.save_path
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    device = args.device

    # Data
    logger.info('==> Preparing data..')
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    _, testset = get_dataset('MNIST_id', download=False, seen=args.seen, train=False)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=1, shuffle=False, num_workers=8)
    # Model
    logger.info('==> Building model..')
    model = ConditionalVAE2(z_dim=32, image_channels=1, class_num=10, device=args.device)
    model.to(device)
    model.load_state_dict(torch.load('/home/linhw/myproject/Attack-Vae/ckpt/cvae_partition1.pkl', map_location=device))
    logger.info("Attack started")

    untargeted_attack(model, testloader, device)

    logger.info("Attack finished.")
    

if __name__ == '__main__':
    main()
