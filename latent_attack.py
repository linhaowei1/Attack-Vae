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

    trainset = torchvision.datasets.MNIST(DATA_PATH, train=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=8)

    testset = torchvision.datasets.MNIST(DATA_PATH, train=False, transform=transform)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=True, num_workers=8)
    # Model
    logger.info('==> Building model..')
    model = VAE()
    model.to(device)
    ## model.load_state_dict(torch.load('/home/linhw/myproject/Attack-Vae/VAE_params.pkl', map_location=device))
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    for epoch in range(args.epochs):
        logger.info("Epoch {} started".format(epoch))

        training_loss, recon_loss, kld_loss = train(model, optimizer, trainloader, device)
        logger.info("training loss = {:.4f}, reconstruction loss = {:.4f}, kld loss = {:.4f}".format(training_loss, recon_loss, kld_loss))

        with torch.no_grad():
            val_loss, recon_loss, kld_loss = val(model, optimizer, testloader, device)
            logger.info("val loss = {:.4f}, reconstruction loss = {:.4f}, kld loss = {:.4f}".format(val_loss, recon_loss, kld_loss))

        if (epoch+1) % 50 == 0:
            sample_images(epoch, model, testloader, device, output_dir)
            model_to_save = model.module if hasattr(model, 'module') else model
            torch.save(model_to_save.state_dict(), '{}/model/{}epoch_model.pt'.format(output_dir, epoch+1))
            logger.info("model saved to {}/model/last_model.pt".format(output_dir))

        logger.info("Epoch {} ended".format(epoch))

    logger.info("Training finished.")
    

if __name__ == '__main__':
    main()
