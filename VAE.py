import torch
from torch import nn
from torch.nn import functional as F
import pdb

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(28*28, 512),  # input layer
            nn.ReLU(),
            nn.Linear(512, 512)
        )
        self.fc2_m = nn.Linear(512, 32)  # get mean
        self.fc2_v = nn.Linear(512, 32)  # get variance
        self.fc3 = nn.Sequential(
            nn.Linear(32, 512),  # asymmetrically connect (input the learned distribution to the decoder)
            nn.ReLU(),
            nn.Linear(512, 512)
        )
        self.fc4 = nn.Linear(512, 28*28)  # ouput layer
        self.relu = F.relu
    #  learn the Gaussian distribution
    def encoder(self, x):
        h = self.relu(self.fc1(x))
        mean = self.fc2_m(h)
        variance = self.fc2_v(h)
        return mean, variance

    # use learned mean and variance to generate code with noise
    def gen_code_with_noise(self, mu, log_var, device):
        temp = torch.exp(log_var / 2).to(device)
        e = torch.randn(temp.size()).to(device)
        return temp * e + mu

    def decoder(self, z):
        h = self.relu(self.fc3(z))
        output = torch.sigmoid(self.fc4(h))  # normalization
        return output

    def forward(self, x, device):
        mu, log_var = self.encoder(x)
        z = self.gen_code_with_noise(mu, log_var, device)
        output = self.decoder(z)
        return output, mu, log_var  # notice that it generate by-product

    def generate(self, x, device, **kwargs):
        return self.forward(x, device)[0].view(-1,1,28,28)

    def sample(self,
               num_samples,
               current_device):
        z = torch.randn(num_samples,self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples.view(-1,1,28,28)