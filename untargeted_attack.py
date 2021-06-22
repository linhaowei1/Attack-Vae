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
from datasets import get_dataset
from tensorboardX import SummaryWriter
from torch.distributions.multivariate_normal import MultivariateNormal
import random
DATA_PATH = '~/data/'

def untargeted_attack(model, dataloader, device, epsilon=0.2, iters=40, alpha=0.2/40):
    model.eval()
    target_img = None
    target_img_label = None
    target_latent = None
    origin_img0 = None
    for idx, (img, label) in tqdm(enumerate(dataloader), total=len(dataloader), desc="attacking"):
        img_flatten = img.view(-1, 28*28).to(device)
        if label[0] == 7:
            origin_img0 = img_flatten[0].data.unsqueeze(0)
            continue
        if origin_img0 is None:
            continue
        tgt_img = origin_img0.repeat(label.size(0),1)
        tgt_img, _, _ = model(tgt_img.to(device), device)
        tgt_img = tgt_img.detach()
        origin_img = img_flatten.data
        for i in range(iters):
            img_flatten.requires_grad = True
            reconstruct, mu, log_var = model(img_flatten, device)  # 通过定义好的神经网络得到预测值
            model.zero_grad()
            # loss
            # reconstruct loss
            reconstruct_loss = F.mse_loss(reconstruct, tgt_img, size_average=False)
            # KL divergence
            kl_divergence = -0.5 * torch.sum(1+log_var-torch.exp(log_var) -mu**2)
            # loss function
            loss = reconstruct_loss + kl_divergence
            # end
            loss.backward()
            adv_images = img_flatten - alpha*img_flatten.grad.sign()
            eta = torch.clamp(adv_images - origin_img, min=-epsilon, max=epsilon)
            img_flatten = torch.clamp(origin_img + eta, min=0, max=1).detach_()
        attack_img, _, _ = model(img_flatten.to(device), device)
        recons_img, _, _ = model(img.view(-1,28*28).to(device), device)
        vutils.save_image(img_flatten.data.view(-1,1,28,28),
                       f'va/{idx}_ori.png',
                       normalize=True,
                       nrow=12)

        vutils.save_image(attack_img.data.view(-1,1,28,28),
                       f'va/{idx}_adv.png',
                       normalize=True,
                       nrow=12)
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
        testset, batch_size=144, shuffle=False, num_workers=8)
    # Model
    logger.info('==> Building model..')
    model = VAE()
    model.to(device)
    model.load_state_dict(torch.load('/home/linhw/myproject/Attack-Vae/checkpoint/model/500epoch_model.pt', map_location=device))
    logger.info("Attack started")

    untargeted_attack(model, testloader, device)

    logger.info("Attack finished.")
    

if __name__ == '__main__':
    main()
