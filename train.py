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
from datasets import get_dataset
from loguru import logger
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
import pandas as pd
import pdb
from tensorboardX import SummaryWriter
import random
DATA_PATH = '~/data/'

def train(model, optimizer, dataloader, device):
    model.train()
    train_loss = 0.0
    reconstruct_loss = 0.0
    kl_divergence = 0.0
    for data, target in dataloader:
        x = data.view(-1, 28*28).to(device)
        y = data.view(-1, 28*28).to(device) # give artificial label y=x
        optimizer.zero_grad()  # 清空上一步的残余更新参数值
        reconstruct, mu, log_var = model(x, device)  # 通过定义好的神经网络得到预测值
        # reconstruct loss
        reconstruct_loss = F.mse_loss(reconstruct, y, size_average=False)
        # KL divergence
        kl_divergence = -0.5 * torch.sum(1+log_var-torch.exp(log_var) -mu**2)
        # loss function
        loss = reconstruct_loss + kl_divergence
        loss.backward()  # back propagation, 更新参数
        optimizer.step()  # 把新参数写入网络
        train_loss = loss.item()*data.size(0)
        # 输出本轮训练结果
    train_loss = train_loss / len(dataloader.dataset)
    reconstruct_loss /= len(dataloader.dataset)
    kl_divergence /=len(dataloader.dataset)
    return train_loss, reconstruct_loss, kl_divergence 

def val(model, optimizer, dataloader, device):
    model.eval()
    train_loss = 0.0
    reconstruct_loss = 0.0
    kl_divergence = 0.0
    for data, target in dataloader:
        x = data.view(-1, 28*28).to(device)
        y = data.view(-1, 28*28).to(device) # give artificial label y=x
        reconstruct, mu, log_var = model(x, device)  # 通过定义好的神经网络得到预测值
        # reconstruct loss
        reconstruct_loss = F.mse_loss(reconstruct, y, size_average=False)
        # KL divergence
        kl_divergence = -0.5 * torch.sum(1+log_var-torch.exp(log_var) -mu**2)
        # loss function
        loss = reconstruct_loss + kl_divergence
        train_loss = loss.item()*data.size(0)
        # 输出本轮训练结果
    train_loss = train_loss / len(dataloader.dataset)
    reconstruct_loss /= len(dataloader.dataset)
    kl_divergence /=len(dataloader.dataset)
    return train_loss, reconstruct_loss, kl_divergence 

def sample_images(epoch, model, dataloader, device, output_dir):
    # Get sample reconstruction image
    model.eval()
    test_input, test_label = next(iter(dataloader))
    test_input = test_input.to(device)
    test_input = test_input.view(-1,28*28)
    test_label = test_label.to(device)
    recons = model.generate(test_input, device)
    vutils.save_image(recons.data,
                        f"{output_dir}/pic/"
                        f"recons_{epoch+1}epoch.png",
                        normalize=True,
                        nrow=12)

    vutils.save_image(test_input.data.view(-1,1,28,28),
                       f"{output_dir}/pic/"
                       f"real_img_{epoch+1}epoch.png",
                       normalize=True,
                       nrow=12)

    try:
        samples = model.sample(144, device, labels = test_label)
        vutils.save_image(samples.cpu().data,
                            f"{output_dir}/pic/"
                            f"sample_{epoch+1}epoch.png",
                            normalize=True,
                            nrow=12)
    except:
        pass
    del test_input, recons #, samples

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
    trainset, testset = get_dataset('MNIST_id', download=False, seen=args.seen, train=False)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=8)
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

        if (epoch+1) % 20 == 0:
            sample_images(epoch, model, testloader, device, output_dir)
            model_to_save = model.module if hasattr(model, 'module') else model
            torch.save(model_to_save.state_dict(), '{}/model/{}epoch_model_seen{}.pt'.format(output_dir, epoch+1, args.seen))
            logger.info("model saved to {}/model/last_model_seen{}.pt".format(output_dir, args.seen))

        logger.info("Epoch {} ended".format(epoch))

    logger.info("Training finished.")
    

if __name__ == '__main__':
    main()
