import os
import torch
import torch.nn as nn
from VAE import VAE
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.nn import DataParallel
from small_cnn import SmallCNN
import torchvision
import torchvision.utils as vutils
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.datasets import ImageFolder
import pdb
DATA_PATH = '~/data/'
import argparse
from tqdm import tqdm
from loguru import logger
from utils import get_args

def generate(vae, model, dataloader, device):
    model.eval()
    labels_cnt = [0] * 10
    with torch.no_grad():
        for batch_idx, (inputs, targets) in tqdm(enumerate(dataloader), total=len(dataloader), desc="testing"):
            inputs = inputs.to(device)
            recon, _, _ = vae(inputs.view(-1,28*28).to(device), device)
            recon_img = recon.view(-1, 1, 28 ,28)
            outputs = model(recon_img.to(device))
            _, predicted = outputs.max(1)
            label = int(targets.item())
            predicted = int(predicted.item())
            if label != predicted:
                label_cnt = labels_cnt[label]
                output_dir = f'hardimgs/{label}'
                if not os.path.exists(output_dir):
                   os.makedirs(output_dir)
                vutils.save_image(torch.cat([inputs.squeeze().cpu(), recon_img.squeeze().cpu()],dim=0),f'{output_dir}/{label_cnt}.png')
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
    best_acc = 0  # best test accuracy

    # Data
    logger.info('==> Preparing data..')
    if args.dataset == 'MNIST':
        transform=transforms.Compose([
            transforms.ToTensor()
        ])
        transform_train = transform
        tramsform_test = transform
    else:
        raise NotImplementedError
    
    epochs = args.epochs
    GEN_PATH = f'/home/linhw/myproject/Attack-Vae/gen_imgs-{epochs}'
    genset = torchvision.datasets.MNIST(DATA_PATH, train=False, transform=transforms.Compose(
        [
            transforms.ToTensor()
        ]
    ))
    genloader = torch.utils.data.DataLoader(genset, batch_size=1, shuffle=False, num_workers=8)
    # Model
    logger.info('==> Building model..')
    if args.model == 'smallCNN':
        model = SmallCNN()
    model = model.to(device)
    model.load_state_dict(torch.load(args.params, map_location=device)['state_dict'])
    vae = VAE()
    vae = vae.to(device)
    vae.load_state_dict(torch.load('/home/linhw/myproject/Attack-Vae/checkpoint/model/500epoch_model.pt', map_location=device))

    generate(vae, model, genloader, device)
    logger.info("generate finished")


if __name__ == '__main__':
    main()
