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

def main():
    args = get_args()
    epochs = args.epochs
    batch_size = args.batch_size
    device = args.device
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    dataset = torchvision.datasets.MNIST(DATA_PATH, train=False, transform=transform)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=8)
    
    model = VAE()
    if args.params == 'None':
        pass
    else:
        model.load_state_dict(torch.load(args.params, map_location=device))
    model.to(device)
    labels_cnt = [0] * 10
    with torch.no_grad():
        model.eval()
        for data, labels in dataloader:
            x = data.view(-1, 28*28).to(device)
            label = int(labels.data)
            reconstruct, _, _ = model(x, device)
            label_cnt = labels_cnt[label]
            output_dir = f'gen_imgs-{epochs}/{label}'
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            vutils.save_image(reconstruct.view(-1, 1, 28, 28), f'gen_imgs-{epochs}/{label}/{label_cnt}.png')
            labels_cnt[label] += 1
    print(labels_cnt)
    
if __name__ == '__main__':
    main()

