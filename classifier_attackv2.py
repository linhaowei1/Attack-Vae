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
    classifier = SmallCNN()
    classifier.to(device)

    classifier.load_state_dict(torch.load('/home/linhw/myproject/Attack-Vae/MNIST_small_cnn.pth', map_location=device)['state_dict'])

    logger.info("Attack started")

    classifier_attackv3(model, classifier, testloader, device)

    logger.info("Attack finished.")
    
def classifier_attackv2(model, classifier, testloader, device, eps=0.2, alpha=0.2/40, iters=40):
    model.eval()
    classifier.eval()
    for idx, (image, label)in enumerate(testloader):
        image, label = image.view(-1,28*28).to(device), label.to(device)
        loss = nn.CrossEntropyLoss()
        ori_image = image.data
        for i in range(iters):
            image.requires_grad = True
            reconstruct, mu, log_var = model(image, device)
            outputs = classifier(reconstruct.view(-1,1,28,28))

            model.zero_grad()
            classifier.zero_grad()

            # ca attack
            cost = F.cross_entropy(outputs, label) 
            cost.backward()
            
            # ca attack 
            adv_image = image + alpha*image.grad.sign()
            # adv_image = image + alpha*image.grad.sign()
            eta = torch.clamp(adv_image - ori_image, min=-eps, max=eps)
            image = torch.clamp(ori_image + eta, min=0, max=1).detach_()   
        
        reconstruct, mu, log_var = model(image, device)
        vutils.save_image(ori_image.data.view(-1,1,28,28),
                       f'uca/{idx}_ori.png',
                       normalize=True,
                       nrow=12)
        vutils.save_image(image.data.view(-1,1,28,28),
                       f'uca/{idx}_pert.png',
                       normalize=True,
                       nrow=12)
        vutils.save_image(reconstruct.data.view(-1,1,28,28),
                       f'uca/{idx}_adv.png',
                       normalize=True,
                       nrow=12)
        
def norm(x):
    norm = torch.norm(x, p=2, dim=1)
    x = x / (norm.expand(1, -1).t() + .0001)
    return x

def classifier_attackv3(model, classifier, testloader, device, iters=5, alpha=.000000000000001):
    model.eval()
    classifier.eval()
    from torch.autograd import grad

    for idx, (image, label)in enumerate(testloader):
        image, label = image.view(-1,28*28).to(device), label.to(device)
        loss = nn.CrossEntropyLoss()
            
        # First encode the image into latent space (z)
        
        mu, logvar = model.encoder(image)
        z_0 = model.gen_code_with_noise(mu, logvar, device)
        z = z_0.clone()

        # Now perform gradient descent to update z
        for i in range(iters):
            # Classify with one extra class
            reconstruct = model.decoder(z)
            logits = classifier(reconstruct.view(-1,1,28,28))
            # Use the extra class as a counterfactual target
            augmented_logits = F.pad(logits, pad=(0,1))
            # Maximize classification probability of the counterfactual target
            batch_size, num_classes = logits.shape
            target_tensor = torch.LongTensor(batch_size).to(device)
            target_tensor[:] = num_classes

        # Maximize classification probability of the counterfactual target
            cf_loss = F.nll_loss(F.log_softmax(augmented_logits, dim=1), target_tensor)
            # Regularize with distance to original z
            distance_loss = torch.mean((z - z_0) ** 2)

            # Move z toward the "open set" class
            loss = cf_loss + distance_loss
            dc_dz = grad(loss, z, loss)[0]
            z = z - alpha * dc_dz
            # Sanity check: Clip gradients to avoid nan in ill-conditioned inputs
            #dc_dz = torch.clamp(dc_dz, -.1, .1)

            # Optional: Normalize to the unit sphere (match encoder's settings)
            z = norm(z)

        reconstruct = model.decoder(z)
        vutils.save_image(image.data.view(-1,1,28,28),
                       f'uca/{idx}_ori.png',
                       normalize=True,
                       nrow=12)

        vutils.save_image(reconstruct.data.view(-1,1,28,28),
                       f'uca/{idx}_adv.png',
                       normalize=True,
                       nrow=12)
if __name__ == '__main__':
    main()
