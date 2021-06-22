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
from small_cnn import SmallCNN
DATA_PATH = '~/data/'
class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(32, 16)
        self.fc2 = nn.Linear(16, 10)
    
    def forward(self, z):
        z = F.relu(self.fc1(z))
        z = self.fc2(z)
        return z
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
    classifier = Classifier()
    classifier.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    logger.info('Triain classifier.')
    for epoch in range(30):
        loss, acc = train_classifier(model, classifier, device, optimizer, testloader)
        logger.info(f'acc = {acc}, loss = {loss}')
    logger.info('Triaining finished.')

    logger.info("Attack started")

    classifier_attack(model, classifier, testloader, device)

    logger.info("Attack finished.")
    
def train_classifier(model, classifier, device, optimizer, dataloader):
    classifier.train()
    model.eval()
    train_loss = 0.0
    acc = 0.0
    total = 0.0
    for image, label in dataloader:
        image,label = image.view(-1,28*28).to(device),label.to(device)
        optimizer.zero_grad()  
        mu, logvar = model.encoder(image) 
        z = model.gen_code_with_noise(mu, logvar, device)
        pred = classifier(z)
        loss = F.cross_entropy(pred, label)
        loss.backward()
        optimizer.step()
        train_loss = loss.item()*label.size(0)
        acc += (pred.max(1)[1] == label).sum().item()
        total += label.size(0)
    acc /= total
    return train_loss, acc

def pgd_attack(model, images, labels, device, eps=0.3, alpha=2/255, iters=40) :
    images = images.view(-1,28*28).to(device)
    labels = labels.to(device)
    loss = nn.CrossEntropyLoss()
        
    ori_images = images.data
        
    for i in range(iters) :    
        images.requires_grad = True
        outputs = model(images)

        model.zero_grad()

        cost = loss(outputs, labels).to(device)
        cost.backward()

        adv_images = images + alpha*images.grad.sign()
        eta = torch.clamp(adv_images - ori_images, min=-eps, max=eps)
        images = torch.clamp(ori_images + eta, min=0, max=1).detach_()           
    return images

def classifier_attack(model, classifier, testloader, device, eps=0.05, alpha=0.05/40, iters=40):
    model.eval()
    classifier.eval()
    for idx, (image, label)in enumerate(testloader):
        image, label = image.view(-1,28*28).to(device), label.to(device)
        loss = nn.CrossEntropyLoss()
        ori_image = image.data
        for i in range(iters):
            image.requires_grad = True
            mu, logvar = model.encoder(image)
            z = model.gen_code_with_noise(mu, logvar, device)
            outputs = classifier(z)

            model.zero_grad()
            classifier.zero_grad()

            # ca attack
            label = torch.tensor([7] * label.size(0)).to(device)
            
            cost = loss(outputs, label).to(device)
            cost.backward()
            
            # ca attack 
            adv_image = image - alpha*image.grad.sign()
            # adv_image = image + alpha*image.grad.sign()
            eta = torch.clamp(adv_image - ori_image, min=-eps, max=eps)
            image = torch.clamp(ori_image + eta, min=0, max=1).detach_()   
        
        reconstruct, mu, log_var = model(image, device)
        vutils.save_image(ori_image.data.view(-1,1,28,28),
                       f'ca/{idx}_ori.png',
                       normalize=True,
                       nrow=12)
        vutils.save_image(image.data.view(-1,1,28,28),
                       f'ca/{idx}_pert.png',
                       normalize=True,
                       nrow=12)
        vutils.save_image(reconstruct.data.view(-1,1,28,28),
                       f'ca/{idx}_adv.png',
                       normalize=True,
                       nrow=12)
        
if __name__ == '__main__':
    main()
