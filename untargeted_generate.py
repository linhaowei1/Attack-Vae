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
import random
DATA_PATH = '~/data/'

def untargeted_generate(model, dataloader, device, epsilon=0.5, iters=30, alpha=1/60):
    model.eval()
    target_img = None
    target_img_label = None
    target_latent = None
    labels_cnt = [0] * 10
    for idx, (img, label) in tqdm(enumerate(dataloader), total=len(dataloader), desc="training"):
        if idx >= int(len(dataloader) / 10):
            break
        img_flatten = img.view(-1, 28*28).to(device)
        origin_img = img_flatten.data
        label = int(label.item())
        label_cnt = labels_cnt[label]
        for i in range(iters):
            img_flatten.requires_grad = True
            reconstruct, mu, log_var = model(img_flatten, device)  # 通过定义好的神经网络得到预测值
            model.zero_grad()
            # loss
            # reconstruct loss
            reconstruct_loss = F.mse_loss(reconstruct, origin_img, size_average=False)
            # KL divergence
            kl_divergence = -0.5 * torch.sum(1+log_var-torch.exp(log_var) -mu**2)
            # loss function
            loss = reconstruct_loss + kl_divergence
            # end
            loss.backward()
            adv_images = img_flatten + alpha*img_flatten.grad.sign()
            eta = torch.clamp(adv_images - origin_img, min=-epsilon, max=epsilon)
            img_flatten = torch.clamp(origin_img + eta, min=0, max=1).detach_()
        attack_img, _, _ = model(img_flatten.to(device), device)
        output_dir = f'untargeted_generate/{label}'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        vutils.save_image(torch.cat([attack_img.view(-1,1,28,28).cpu()],dim=0),f'{output_dir}/{label_cnt}.png')
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
        testset, batch_size=1, shuffle=True, num_workers=8)
    # Model
    logger.info('==> Building model..')
    model = VAE()
    model.to(device)
    model.load_state_dict(torch.load('/home/linhw/myproject/Attack-Vae/checkpoint/model/500epoch_model.pt', map_location=device))
    logger.info("Attack started")

    untargeted_generate(model, testloader, device)

    logger.info("Attack finished.")
    

if __name__ == '__main__':
    main()
