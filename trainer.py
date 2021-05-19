
import torch
from torchvision import datasets
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt

# define hyper parameters
n_epochs = 50
batch_size = 64
learning_rate = 0.001

# load data
train_data = datasets.MNIST(
    root = './data', train=True, download=True, transform=transforms.ToTensor())
test_data = datasets.MNIST(
    root = './data', train=False, download=True, transform=transforms.ToTensor())

# create the loader
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, num_workers=0)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, num_workers=0)


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(28*28, 512),  # input layer
            nn.ReLU(),
            nn.Linear(512, 512)
        )
        self.fc2_m = nn.Linear(400, 32)  # get mean
        self.fc2_v = nn.Linear(400, 32)  # get variance
        self.fc3 = nn.Sequential(
            nn.Linear(32, 512),  # asymmetrically connect (input the learned distribution to the decoder)
            nn.ReLU(),
            nn.Linear(512, 512)
        )
        self.fc4 = nn.Linear(512, 28*28)  # ouput layer

    #  learn the Gaussian distribution
    def encoder(self, x):
        h = F.relu(self.fc1(x))
        mean = self.fc2_m(h)
        variance = self.fc2_v(h)
        return mean, variance

    # use learned mean and variance to generate code with noise
    def gen_code_with_noise(self, mu, log_var):
        temp = torch.exp(log_var / 2)
        e = torch.randn(temp.size())
        return temp * e + mu

    def decoder(self, z):
        h = F.relu(self.fc3(z))
        output = torch.sigmoid(self.fc4(h))  # normalization
        return output

    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = self.gen_code_with_noise(mu, log_var)
        output = self.decoder(z)
        return output, mu, log_var  # notice that it generate by-product

model = VAE()