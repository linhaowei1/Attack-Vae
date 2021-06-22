import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from models.conv.nets import ConvLayers, DeconvLayers, ConvLayers2, DeconvLayers2
from models.conv.deeper_cnn import classifier32
from models.fc.nets import MLP
from models.utils import modules
from models.utils import loss_functions as lf
from models.utils.transformHelper import rot_img, cut_perm
import pdb


class mainModel(nn.Module):
    """
    feature_extractor(CNN) -> classifier (MLP)
    """

    def __init__(self, image_size, image_channels, classes, fc_latent_dim=512):
        super(mainModel, self).__init__()
        self.image_size = image_size
        self.image_channels = image_channels
        self.classes = classes

        # for encoder
        # self.convE = ConvLayers2(image_channels)
        self.convE = classifier32(image_channels=image_channels)
        self.flatten = modules.Flatten()

        # classifier
        self.classifier = MLP(self.convE.out_feature_dim, classes)

        self.optimizer = None  # needs to be set before training starts

        self.device = None  # needs to be set before using the model

    # --------- FROWARD FUNCTIONS ---------#
    def encode(self, x):
        """
        pass input through feed-forward connections to get [image_features]
        """
        # Forward-pass through conv-layers
        hidden_x = self.convE(x)

        return hidden_x

    def classify(self, x):
        """
        For input [x] (image or extracted "internal“ image features),
        return predicted scores (<2D tensor> [batch_size] * [classes])
        """
        result = self.classifier(x)
        return result

    def forward(self, x):
        """
        Forward function to propagate [x] through the encoder and the classifier.

                Input: - [x]          <4D-tensor> of shape [batch_size]x[channels]x[image_size]x[image_size]
                Output - [prediction]

        """
        hidden_x = self.encode(x)
        prediction = self.classifier(hidden_x)
        return prediction

    # ------------------TRAINING FUNCTIONS----------------------#
    def train_a_batch(self, x, y, epoch_num, batch_num):
        """
        Train model for one batch ([x], [y])
        """
        # Set model to training-mode
        self.train()
        for p in self.convE.parameters():
            p.requires_grad = True

        # Reset optimizer
        self.optimizer.zero_grad()

        # Run the model
        hidden_x = self.encode(x)
        prediction = self.classifier(hidden_x)
        predL = F.cross_entropy(prediction, y, reduction='none')
        loss = lf.weighted_average(predL, weights=None, dim=0)

        loss.backward()

        self.optimizer.step()  # 更新梯度写回

        if batch_num % 100 == 0:
            print(
                'Epoch {}, Batch index {}, loss = {:.6f}'.format(
                    epoch_num, batch_num, loss.item()))

        return loss.item()
