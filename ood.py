import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.nn import DataParallel
from small_cnn import SmallCNN
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.datasets import ImageFolder
import pdb
DATA_PATH = '~/data/'
import argparse
from tqdm import tqdm
from loguru import logger
from utils import get_args
import numpy as np
from sklearn import metrics

def ood_detect(model, dataloader, device, args):
    model.eval()
    label = []
    score = []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in tqdm(enumerate(dataloader), total=len(dataloader), desc="testing"):
            inputs = inputs.to(device)
            label += targets.cpu().numpy().tolist()
            outputs = F.softmax(model(inputs), dim=1)
            _score, _ = outputs.max(1)

            if args.mode == 'adv_openset':
                _score = _score-outputs[:,6]
            elif args.mode == 'adv_datasets':
                _score = -outputs[:,10]

            score += _score.cpu().numpy().tolist()
    return score,label

def my_get_subclass_dataset(dataset, classes):
    if not isinstance(classes, list):
        classes = [classes]
    indices = []
    for idx, data in enumerate(dataset):
        if data[1] in classes:
            indices.append(idx)

    dataset = torch.utils.data.dataset.Subset(dataset, indices)
    return dataset


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
    transform=transforms.Compose([
        transforms.ToTensor()
    ])
    transform_train = transform
    tramsform_test = transform

    if args.mode == 'openset' or args.mode == 'adv_openset':
        testset = torchvision.datasets.MNIST(DATA_PATH, train=False, transform=transform)    
        testset = my_get_subclass_dataset(testset, [0,1,2,3,4,5])
        oodset = torchvision.datasets.MNIST(DATA_PATH, train=False, transform=transform)
        oodset = my_get_subclass_dataset(oodset, [6,7,8,9])
    elif args.mode == 'adv_datasets':
        testset = torchvision.datasets.MNIST(DATA_PATH, train=False, transform=transform, target_transform= lambda x : 0)    
        oodset = torchvision.datasets.FashionMNIST(DATA_PATH, train=False, transform=transform, target_transform= lambda x:1)
    else:
        testset = torchvision.datasets.MNIST(DATA_PATH, train=False, transform=transform, target_transform= lambda x : 0)    
        oodset = torchvision.datasets.FashionMNIST(DATA_PATH, train=False, transform=transform, target_transform= lambda x:1)
    dataset = testset + oodset
    testloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8)

    logger.info('==> Building model..')
    model = SmallCNN()
    if args.mode == 'openset':
        model.classifier[-1] = nn.Linear(200, 6)
        model.load_state_dict(torch.load('/home/linhw/myproject/Attack-Vae/checkpoint/012345_ood_model.pt', map_location=device))
    elif args.mode == 'adv_openset':
        model.classifier[-1] = nn.Linear(200, 7)
        model.load_state_dict(torch.load('/home/linhw/myproject/Attack-Vae/checkpoint/012345_adv_ood_model.pt', map_location=device))
    elif args.mode == 'adv_datasets':
        model.classifier[-1] = nn.Linear(200, 11)
        model.load_state_dict(torch.load('/home/linhw/myproject/Attack-Vae/checkpoint/123456789model.pt', map_location=device))    
    else:
        model.load_state_dict(torch.load('MNIST_small_cnn.pth', map_location=device)['state_dict'])
    model = model.to(device)
    score, label = ood_detect(model, testloader, device, args)
    if args.mode == 'openset' or args.mode == 'adv_openset':
        label = [la >= 6 for la in label]

    fpr, tpr, thresholds = metrics.roc_curve(label, score, pos_label=0)
    auc = metrics.auc(fpr, tpr)
    
    logger.info(f"test finished, auc = {auc}")

if __name__ == '__main__':
    main()
