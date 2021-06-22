import torch
from torch import nn
import numpy as np
import torch.nn.functional as F


class ConvLayers(nn.Module):
    """
    Convolutional feature extractor model for (natural) images.

    Input:  [batch_size] x [image_channels] x [image_size] x [image_size] tensor
    Output: [batch_size] x [out_channels] x [out_size] x [out_size] tensor
                - out_channels = 128
                - out_size = image_size
    """
    def __init__(self, image_channels):
        super(ConvLayers, self).__init__()
        self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=(4, 4), stride=2, padding=(15, 15))
        # [image_channels, image_size, image_size] -> [64, image_size, image_size]
        self.conv2 = nn.Conv2d(64, 128, kernel_size=(4, 4), stride=2, padding=(15, 15))
        # [32, image_size, image_size] -> [128, image_size, image_size]
        self.out_channels = 128
        self.out_feature_dim = 128 * 28 * 28

    def forward(self, x):
        x = F.elu(self.conv1(x))
        feature = F.elu(self.conv2(x))

        return feature


class DeconvLayers(nn.Module):
    """
    "Deconvolutional" feature decoder model for (natural) images.

    Input:  [batch_size] x [in_channels] x [in_size] x [in_size] tensor
    Output: [batch_size] x [image_channels] x [final_size] x [final_size] tensor

    """
    def __init__(self, image_channels):
        super(DeconvLayers, self).__init__()
        self.image_channels = image_channels
        self.in_feature_dim = 7 * 7 * 128
        self.deconv1 = nn.ConvTranspose2d(
            in_channels=128, out_channels=64, kernel_size=4, padding=1, stride=2)
        self.deconv2 = nn.ConvTranspose2d(
            in_channels=64, out_channels=self.image_channels, kernel_size=4, padding=1, stride=2)

    def forward(self, x):
        x = F.elu(self.deconv1(x))
        x = torch.sigmoid(self.deconv2(x))

        return x


class ConvLayers2(nn.Module):
    """
    Convolutional feature extractor model for (natural) images.

    Input:  [batch_size] x [image_channels] x [image_size] x [image_size] tensor
    Output: [batch_size] x [out_channels] x [out_size] x [out_size] tensor
                - out_channels = 256
                - out_size = 4
    """
    def __init__(self, image_channels):
        super(ConvLayers2, self).__init__()
        self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=(3, 3), stride=2, padding=(1, 1))
        # [image_channels, image_size, image_size] -> [64, image_size/2, image_size/2]
        self.conv2 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=2, padding=(1, 1))
        # [64, image_size/2, image_size/2] -> [128, image_size/4, image_size/4]
        self.conv3 = nn.Conv2d(128, 256, kernel_size=(3, 3), stride=2, padding=(1, 1))
        # [128, image_size/4, image_size/4] -> [256, 4, 4]
        self.out_channels = 256
        self.out_feature_dim = 256 * 4 * 4

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        feature = F.relu(self.conv3(x))

        return feature


class DeconvLayers2(nn.Module):
    """
    "Deconvolutional" feature decoder model for (natural) images.

    Input:  [batch_size] x [in_channels] x [in_size] x [in_size] tensor
    Output: [batch_size] x [image_channels] x [final_size] x [final_size] tensor

    """
    def __init__(self, image_channels):
        super(DeconvLayers2, self).__init__()
        self.image_channels = image_channels
        self.in_feature_dim = 4 * 4 * 256
        self.deconv1 = nn.ConvTranspose2d(
            in_channels=256, out_channels=128, kernel_size=3, padding=1, stride=2)
        self.deconv2 = nn.ConvTranspose2d(
            in_channels=128, out_channels=64, kernel_size=4, padding=1, stride=2)
        self.deconv3 = nn.ConvTranspose2d(
            in_channels=64, out_channels=self.image_channels, kernel_size=4, padding=1, stride=2)

    def forward(self, x):
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = torch.sigmoid(self.deconv3(x))

        return x