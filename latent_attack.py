from utils import get_args
from VAE import VAE
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.utils as vutils
import math
import os
from loguru import logger
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
import pandas as pd
import pdb
from tensorboardX import SummaryWriter
from torch.distributions.multivariate_normal import MultivariateNormal
import random
DATA_PATH = '~/data/'
def KL(p_mean, p_logvar, q_mean, q_logvar):
    term1 = p_logvar.sum(dim=1) + q_logvar.sum(dim=1)
    term2 = torch.sum(p_logvar.exp()/q_logvar.exp(), dim=1)
    term3 = torch.sum((p_mean-q_mean) * (p_mean - q_mean)/q_logvar.exp(), dim=1)
    return (term1 + term2 + term3 - p_mean.size(1)) * 0.5

def latent_attack(model, dataloader, device, epsilon=0.3, iters=40, alpha=3/255):
    model.eval()
    target_img = None
    target_img_label = None
    target_latent = None
    labels_cnt = [0] * 10
    for idx, (img, label) in tqdm(enumerate(dataloader), total=len(dataloader), desc="training"):
        if idx == 0 or int(target_img_label) == int(label):
            target_img = img
            target_img_label = label
            mean_t, logvar_t = model.encoder(img.view(-1,28*28).to(device))
        elif int(label) is not int(target_img_label):
            img_flatten = img.view(-1, 28*28).to(device)
            origin_img = img_flatten.data
            label = int(label.item())
            label_cnt = labels_cnt[label]
            for i in range(iters):
                img_flatten.requires_grad = True
                mean, logvar = model.encoder(img_flatten)
                model.zero_grad()
                # loss
                term1 = -logvar.sum(dim=1) + logvar_t.sum(dim=1)
                term2 = torch.sum(logvar.exp()/logvar_t.exp(), dim=1)
                term3 = torch.sum((mean-mean_t) * (mean - mean_t)/logvar_t.exp(), dim=1)
                kld = (term1 + term2 + term3 - mean.size(1)) * 0.5
                # end
                kld.backward(retain_graph=True)
                adv_images = img_flatten + alpha*img_flatten.grad.sign()
                eta = torch.clamp(adv_images - origin_img, min=-epsilon, max=epsilon)
                img_flatten = torch.clamp(origin_img + eta, min=0, max=1).detach_()
            attack_img, _, _ = model(img_flatten.to(device), device)
            recons_img, _, _ = model(img.view(-1,28*28).to(device), device)
            output_dir = f'latent_attack/{label}'
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            vutils.save_image(torch.cat([img.cpu(), img_flatten.view(-1,1,28,28).cpu(), recons_img.view(-1,1,28,28).cpu(), attack_img.view(-1,1,28,28).cpu()],dim=0),f'{output_dir}/{label_cnt}.png')
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

    testset = torchvision.datasets.MNIST(DATA_PATH, train=False, transform=transform)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=1, shuffle=False, num_workers=8)
    # Model
    logger.info('==> Building model..')
    model = VAE()
    model.to(device)
    model.load_state_dict(torch.load('/home/linhw/myproject/Attack-Vae/checkpoint/model/500epoch_model.pt', map_location=device))
    logger.info("Attack started")

    latent_attack(model, testloader, device)

    logger.info("Attack finished.")
    

if __name__ == '__main__':
    main()
