import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from models.conv.nets import ConvLayers, DeconvLayers, ConvLayers2, DeconvLayers2
from models.fc.nets import MLP
from models.utils import modules
from models.utils import loss_functions as lf
from models.utils.transformHelper import rot_img, cut_perm
import pdb


class ConditionalVAE2(nn.Module):
    def __init__(self, z_dim=32, image_channels=1, class_num=10, device='cuda:1'):
        super(ConditionalVAE2, self).__init__()
        self.class_num = class_num

        self.convE = ConvLayers2(image_channels)
        self.flatten = modules.Flatten()
        self.fcE = nn.Linear(self.convE.out_feature_dim, 1024)
        self.z_dim = z_dim
        self.fcE_mean = nn.Linear(1024, self.z_dim)
        self.fcE_logvar = nn.Linear(1024, self.z_dim)
        self.fromZ = nn.Linear(2 * self.z_dim, 1024)
        # self.fromZ = nn.Linear(self.z_dim, 1024)
        self.convD = DeconvLayers2(image_channels)
        self.fcD = nn.Linear(1024, self.convD.in_feature_dim)
        self.to_image = modules.Reshape(image_channels=self.convE.out_channels)
        self.device = device

        self.class_embed = nn.Linear(class_num, self.z_dim)

    def encode(self, x):
        hidden_x = self.convE(x)
        feature = self.flatten(hidden_x)

        hE = F.relu(self.fcE(feature))

        z_mean = self.fcE_mean(hE)
        z_logvar = self.fcE_logvar(hE)

        return z_mean, z_logvar, hE, hidden_x

    def reparameterize(self, mu, logvar):
        # std = 0.1 * torch.exp(logvar).cpu()  # "wrong" version
        std = torch.exp(0.5 * logvar).to(self.device)
        z = torch.randn(std.size()).to(self.device) * std + mu.to(self.device)
        return z

    def decode(self, z, y_embed):
        z =torch.cat([z, y_embed], dim=1) # add label information
        hD = F.relu(self.fromZ(z))
        feature = self.fcD(hD)
        image_recon = self.convD(feature.view(-1, 256, 4, 4))

        return image_recon

    def forward(self, x, y_tensor):
        mu, logvar, hE, hidden_x = self.encode(x)
        z = self.reparameterize(mu, logvar)
        y_embed = self.class_embed(y_tensor)
        x_recon = self.decode(z, y_embed)
        return mu, logvar, x_recon