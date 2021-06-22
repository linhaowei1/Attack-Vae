from PIL import ImageStat
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import torchvision
import argparse
import imutil
from logutil import TimeSeries
from datasets import get_dataset

GENERATIVE_EPOCHS = 30

BATCH_SIZE = 144
LATENT_SIZE = 20
NUM_CLASSES = 6
# Utilities for noise generation, clamping etc
import torch
import time
import numpy as np
from torch.autograd import Variable


def make_noise(batch_size, latent_size, scale, fixed_seed=None):
    noise_t = torch.FloatTensor(batch_size, latent_size * scale * scale)
    if fixed_seed is not None:
        seed(fixed_seed)
    noise_t.normal_(0, 1)
    noise = Variable(noise_t).cuda()
    result = clamp_to_unit_sphere(noise, scale**2)
    if fixed_seed is not None:
        seed(int(time.time()))
    return result


def seed(val=42):
    np.random.seed(val)
    torch.manual_seed(val)
    torch.cuda.manual_seed(val)


# TODO: Merge this with the more fully-featured make_noise()
def gen_noise(K, latent_size):
    noise = torch.zeros((K, latent_size))
    noise.normal_(0, 1)
    noise = clamp_to_unit_sphere(noise)
    return noise


def clamp_to_unit_sphere(x, components=1):
    # If components=4, then we normalize each quarter of x independently
    # Useful for the latent spaces of fully-convolutional networks
    batch_size, latent_size = x.shape
    latent_subspaces = []
    for i in range(components):
        step = latent_size // components
        left, right = step * i, step * (i+1)
        subspace = x[:, left:right].clone()
        norm = torch.norm(subspace, p=2, dim=1)
        subspace = subspace / norm.expand(1, -1).t()  # + epsilon
        latent_subspaces.append(subspace)
    # Join the normalized pieces back together
    return torch.cat(latent_subspaces, dim=1)


def random_seed(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    torch.cuda.manual_seed(seed_value)
    # torch.set_deterministic(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
import random
import numpy as np
import torch
import os
import torch
from torch import nn

def weights_init(m):
    classname = m.__class__.__name__
    # TODO: what about fully-connected layers?
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.05)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class encoder32(nn.Module):
    def __init__(self, latent_size=100, num_classes=2, batch_size=64, **kwargs):
        super(self.__class__, self).__init__()
        self.batch_size = batch_size
        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(1,       64,     3, 1, 1, bias=False)
        self.conv2 = nn.Conv2d(64,      64,     3, 1, 1, bias=False)
        self.conv3 = nn.Conv2d(64,     128,     3, 2, 1, bias=False)

        self.conv4 = nn.Conv2d(128,    128,     3, 1, 1, bias=False)
        self.conv5 = nn.Conv2d(128,    128,     3, 1, 1, bias=False)
        self.conv6 = nn.Conv2d(128,    128,     3, 2, 1, bias=False)
        # Shortcut out of the network at 8x8
        self.conv_out_6 = nn.Conv2d(128, latent_size, 3, 1, 1, bias=False)

        self.conv7 = nn.Conv2d(128,    128,     3, 1, 1, bias=False)
        self.conv8 = nn.Conv2d(128,    128,     3, 1, 1, bias=False)
        self.conv9 = nn.Conv2d(128,    128,     3, 2, 1, bias=False)
        # Shortcut out of the network at 4x4
        self.conv_out_9 = nn.Conv2d(128, latent_size, 3, 1, 1, bias=False)

        self.conv10 = nn.Conv2d(128,    128,     3, 2, 1, bias=False)
        self.conv_out_10 = nn.Conv2d(128, latent_size, 3, 1, 1, bias=False)

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)

        self.bn4 = nn.BatchNorm2d(128)
        self.bn5 = nn.BatchNorm2d(128)
        self.bn6 = nn.BatchNorm2d(128)

        self.bn7 = nn.BatchNorm2d(128)
        self.bn8 = nn.BatchNorm2d(128)
        self.bn9 = nn.BatchNorm2d(128)
        self.bn10 = nn.BatchNorm2d(128)

        self.fc1 = nn.Linear(128*2*2, latent_size)

        self.dr1 = nn.Dropout2d(0.2)
        self.dr2 = nn.Dropout2d(0.2)
        self.dr3 = nn.Dropout2d(0.2)
        self.dr4 = nn.Dropout2d(0.2)

        self.apply(weights_init)
        self.cuda()

    def forward(self, x, output_scale=1):
        batch_size = len(x)
        x = self.dr1(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = nn.LeakyReLU(0.2)(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = nn.LeakyReLU(0.2)(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = nn.LeakyReLU(0.2)(x)

        x = self.dr2(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = nn.LeakyReLU(0.2)(x)
        x = self.conv5(x)
        x = self.bn5(x)
        x = nn.LeakyReLU(0.2)(x)
        x = self.conv6(x)
        x = self.bn6(x)
        x = nn.LeakyReLU(0.2)(x)

        # Image representation is now 8 x 8
        if output_scale == 8:
            x = self.conv_out_6(x)
            x = x.view(batch_size, -1)
            x = clamp_to_unit_sphere(x, 8*8)
            return x

        x = self.dr3(x)
        x = self.conv7(x)
        x = self.bn7(x)
        x = nn.LeakyReLU(0.2)(x)
        x = self.conv8(x)
        x = self.bn8(x)
        x = nn.LeakyReLU(0.2)(x)
        x = self.conv9(x)
        x = self.bn9(x)
        x = nn.LeakyReLU(0.2)(x)

        # Image representation is now 4x4
        if output_scale == 4:
            x = self.conv_out_9(x)
            x = x.view(batch_size, -1)
            x = clamp_to_unit_sphere(x, 4*4)
            return x

        x = self.dr4(x)
        x = self.conv10(x)
        x = self.bn10(x)
        x = nn.LeakyReLU(0.2)(x)

        # Image representation is now 2x2
        if output_scale == 2:
            x = self.conv_out_10(x)
            x = x.view(batch_size, -1)
            x = clamp_to_unit_sphere(x, 2*2)
            return x

        x = x.view(batch_size, -1)
        x = self.fc1(x)
        x = clamp_to_unit_sphere(x)
        return x


class generator32(nn.Module):
    def __init__(self, latent_size=100, batch_size=64, **kwargs):
        super(self.__class__, self).__init__()
        self.latent_size = latent_size
        self.fc1 = nn.Linear(latent_size, 512*2*2, bias=False)

        self.conv2_in = nn.ConvTranspose2d(latent_size, 512, 1, stride=1, padding=0, bias=False)
        self.conv2 = nn.ConvTranspose2d(   512,      512, 4, stride=2, padding=1, bias=False)
        self.conv3_in = nn.ConvTranspose2d(latent_size, 512, 1, stride=1, padding=0, bias=False)
        self.conv3 = nn.ConvTranspose2d(   512,      256, 4, stride=2, padding=1, bias=False)
        self.conv4_in = nn.ConvTranspose2d(latent_size, 256, 1, stride=1, padding=0, bias=False)
        self.conv4 = nn.ConvTranspose2d(   256,      128, 4, stride=2, padding=1, bias=False)
        self.conv5 = nn.ConvTranspose2d(   128,        1, 4, stride=2, padding=3)

        self.bn1 = nn.BatchNorm2d(512)
        self.bn2 = nn.BatchNorm2d(512)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm2d(128)

        self.batch_size = batch_size
        self.apply(weights_init)
        self.cuda()

    def forward(self, x, input_scale=1):
        batch_size = x.shape[0]
        if input_scale <= 1:
            x = self.fc1(x)
            x = x.resize(batch_size, 512, 2, 2)

        # 512 x 2 x 2
        if input_scale == 2:
            x = x.view(batch_size, self.latent_size, 2, 2)
            x = self.conv2_in(x)
        if input_scale <= 2:
            x = self.conv2(x)
            x = nn.LeakyReLU()(x)
            x = self.bn2(x)

        # 512 x 4 x 4
        if input_scale == 4:
            x = x.view(batch_size, self.latent_size, 4, 4)
            x = self.conv3_in(x)
        if input_scale <= 4:
            x = self.conv3(x)
            x = nn.LeakyReLU()(x)
            x = self.bn3(x)

        # 256 x 8 x 8
        if input_scale == 8:
            x = x.view(batch_size, self.latent_size, 8, 8)
            x = self.conv4_in(x)
        if input_scale <= 8:
            x = self.conv4(x)
            x = nn.LeakyReLU()(x)
            x = self.bn4(x)
        # 128 x 16 x 16
        x = self.conv5(x)
        # 3 x 32 x 32
        x = nn.Sigmoid()(x)
        return x


class multiclassDiscriminator32(nn.Module):
    def __init__(self, latent_size=100, num_classes=2, batch_size=64, **kwargs):
        super(self.__class__, self).__init__()
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(1,       64,     3, 1, 1, bias=False)
        self.conv2 = nn.Conv2d(64,      64,     3, 1, 1, bias=False)
        self.conv3 = nn.Conv2d(64,     128,     3, 2, 1, bias=False)

        self.conv4 = nn.Conv2d(128,    128,     3, 1, 1, bias=False)
        self.conv5 = nn.Conv2d(128,    128,     3, 1, 1, bias=False)
        self.conv6 = nn.Conv2d(128,    128,     3, 2, 1, bias=False)

        self.conv7 = nn.Conv2d(128,    128,     3, 1, 1, bias=False)
        self.conv8 = nn.Conv2d(128,    128,     3, 1, 1, bias=False)
        self.conv9 = nn.Conv2d(128,    128,     3, 2, 1, bias=False)

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)

        self.bn4 = nn.BatchNorm2d(128)
        self.bn5 = nn.BatchNorm2d(128)
        self.bn6 = nn.BatchNorm2d(128)

        self.bn7 = nn.BatchNorm2d(128)
        self.bn8 = nn.BatchNorm2d(128)
        self.bn9 = nn.BatchNorm2d(128)

        self.fc1 = nn.Linear(128*4*4 * 2, num_classes)
        self.dr1 = nn.Dropout2d(0.2)
        self.dr2 = nn.Dropout2d(0.2)
        self.dr3 = nn.Dropout2d(0.2)

        self.apply(weights_init)
        self.cuda()

    def forward(self, x, return_features=False):
        batch_size = len(x)
        x = self.dr1(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = nn.LeakyReLU(0.2)(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = nn.LeakyReLU(0.2)(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = nn.LeakyReLU(0.2)(x)

        x = self.dr2(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = nn.LeakyReLU(0.2)(x)
        x = self.conv5(x)
        x = self.bn5(x)
        x = nn.LeakyReLU(0.2)(x)
        x = self.conv6(x)
        x = self.bn6(x)
        x = nn.LeakyReLU(0.2)(x)

        x = self.dr3(x)
        x = self.conv7(x)
        x = self.bn7(x)
        x = nn.LeakyReLU(0.2)(x)
        x = self.conv8(x)
        x = self.bn8(x)
        x = nn.LeakyReLU(0.2)(x)
        x = self.conv9(x)
        x = self.bn9(x)
        x = nn.LeakyReLU(0.2)(x)

        x = x.view(batch_size, -1)
        if return_features:
            return x

        # Lazy minibatch discrimination: avg of other examples' features
        batch_avg = torch.exp(-x.mean(dim=0))
        batch_avg = batch_avg.expand(batch_size, -1)
        x = torch.cat([x, batch_avg], dim=1)
        x = self.fc1(x)
        return x


class classifier32(nn.Module):
    def __init__(self, num_classes=6, batch_size=256, **kwargs):
        super(self.__class__, self).__init__()
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(1, 64, 3, 1, 1, bias=False)
        self.conv2 = nn.Conv2d(64, 64, 3, 1, 1, bias=False)
        self.conv3 = nn.Conv2d(64, 128, 3, 2, 1, bias=False)

        self.conv4 = nn.Conv2d(128, 128, 3, 1, 1, bias=False)
        self.conv5 = nn.Conv2d(128, 128, 3, 1, 1, bias=False)
        self.conv6 = nn.Conv2d(128, 128, 3, 2, 1, bias=False)

        self.conv7 = nn.Conv2d(128, 128, 3, 1, 1, bias=False)
        self.conv8 = nn.Conv2d(128, 128, 3, 1, 1, bias=False)
        self.conv9 = nn.Conv2d(128, 128, 3, 2, 1, bias=False)

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)

        self.bn4 = nn.BatchNorm2d(128)
        self.bn5 = nn.BatchNorm2d(128)
        self.bn6 = nn.BatchNorm2d(128)

        self.bn7 = nn.BatchNorm2d(128)
        self.bn8 = nn.BatchNorm2d(128)
        self.bn9 = nn.BatchNorm2d(128)

        self.fc1 = nn.Linear(128*4*4, num_classes)
        self.dr1 = nn.Dropout2d(0.2)
        self.dr2 = nn.Dropout2d(0.2)
        self.dr3 = nn.Dropout2d(0.2)

        self.apply(weights_init)
        self.cuda()

    def forward(self, x, return_features=False):
        batch_size = len(x)

        x = self.dr1(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = nn.LeakyReLU(0.2)(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = nn.LeakyReLU(0.2)(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = nn.LeakyReLU(0.2)(x)

        x = self.dr2(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = nn.LeakyReLU(0.2)(x)
        x = self.conv5(x)
        x = self.bn5(x)
        x = nn.LeakyReLU(0.2)(x)
        x = self.conv6(x)
        x = self.bn6(x)
        x = nn.LeakyReLU(0.2)(x)

        x = self.dr3(x)
        x = self.conv7(x)
        x = self.bn7(x)
        x = nn.LeakyReLU(0.2)(x)
        x = self.conv8(x)
        x = self.bn8(x)
        x = nn.LeakyReLU(0.2)(x)
        x = self.conv9(x)
        x = self.bn9(x)
        x = nn.LeakyReLU(0.2)(x)

        x = x.view(batch_size, -1)
        if return_features:
            return x
        x = self.fc1(x)
        return x

class Classifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_classes)
        self.cuda()

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x


class Encoder(nn.Module):
    def __init__(self, latent_size):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, latent_size)
        self.cuda()

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        x = norm(x)
        return x


# Project to the unit sphere
def norm(x):
    norm = torch.norm(x, p=2, dim=1)
    x = x / (norm.expand(1, -1).t() + .0001)
    return x


class Generator(nn.Module):
    def __init__(self, latent_size):
        super().__init__()
        self.fc1 = nn.Linear(latent_size, 128)
        self.fc2 = nn.Linear(128, 196)
        self.conv1 = nn.ConvTranspose2d(4, 32, stride=2, kernel_size=4, padding=1)
        self.conv2 = nn.ConvTranspose2d(32, 1, stride=2, kernel_size=4, padding=1)
        self.cuda()

    def forward(self, x):
        x = self.fc1(x)
        x = F.leaky_relu(x, 0.2)
        x = self.fc2(x)
        x = F.leaky_relu(x, 0.2)
        x = x.view(-1, 4, 7, 7)
        x = self.conv1(x)
        x = F.leaky_relu(x, 0.2)
        x = self.conv2(x)
        x = torch.sigmoid(x)
        return x


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 1)
        self.cuda()

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def openset_fuxin(dataloader, netC):
    openset_scores = []
    for i, (images, labels) in enumerate(dataloader):
        images = Variable(images, volatile=True)
        logits = netC(images.cuda())
        augmented_logits = F.pad(logits, pad=(0,1))
        # The implicit K+1th class (the open set class) is computed
        #  by assuming an extra linear output with constant value 0
        preds = F.softmax(augmented_logits)
        #preds = augmented_logits
        prob_unknown = preds[:, -1]
        prob_known = preds[:, :-1].max(dim=1)[0]
        prob_open = prob_unknown - prob_known

        openset_scores.extend(prob_open.data.cpu().numpy())
    return np.array(openset_scores)


def train_openset_classifier(dataloader, aux_dataset, netC):

    optimizerC = torch.optim.Adam(netC.parameters())
    for (images, labels), aux_images in zip(dataloader, aux_dataset):
        images = images.cuda()
        labels = labels.cuda()
        ############################
        # Classifier Update
        ############################
        netC.zero_grad()

        # Classify real examples into the correct K classes
        classifier_logits = netC(images)
        augmented_logits = F.pad(classifier_logits, (0,1))
        errC = F.cross_entropy(augmented_logits, labels)
        errC.backward()

        # Classify aux_dataset examples as open set
        classifier_logits = netC((aux_images).cuda())
        augmented_logits = F.pad(classifier_logits, (0,1))
        log_soft_open = F.log_softmax(augmented_logits, dim=1)[:, -1]
        errOpenSet = -log_soft_open.mean()
        errOpenSet.backward()

        optimizerC.step()
        ############################

        #log.collect_prediction('Classifier Accuracy', netC(images), labels)
    return True

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--partition', type=str, default='012345', help='partition')
    args = parser.parse_args()
    random_seed(args.seed)
    device = 'cuda'
    import time
    if args.partition == '012345':
        partition = 1
    elif args.partition == '123456':
        partition = 6
    elif args.partition == '234567':
        partition = 2
    elif args.partition == '345678':
        partition = 7
    elif args.partition == '456789':
        partition = 3
    
    with open(f'gan_{args.seed}_{args.partition}.txt', 'a') as f:
        f.write('start train\n')
        f.write(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
        f.write('\n')
        # Train and test a perfectly normal, ordinary classifier
        classifier = classifier32(num_classes=NUM_CLASSES)
        classifier.load_state_dict(torch.load(f'/home/linhw/myproject/openmax/ckpt_deeper/6slots_classifier_epoch29_partition1.pkl', map_location=device))
        '''
        for i in range(CLASSIFIER_EPOCHS):
            train_classifier(classifier, load_training_dataset(args))
            test_open_set_performance(classifier)
        '''
        # Build a generative model
        encoder = encoder32(latent_size=LATENT_SIZE)
        generator = generator32(latent_size=LATENT_SIZE)
        discriminator = multiclassDiscriminator32()
        '''
        for i in range(GENERATIVE_EPOCHS):
            encoder.train()
            generator.train()
            discriminator.train()
            train_generative_model(encoder, generator, discriminator, load_training_dataset(args))
        torch.save(encoder.state_dict(), '{}/encoder/seen{}_model_seed{}.pt'.format('checkpoint', args.partition, args.seed))
        torch.save(generator.state_dict(), '{}/generator/seen{}_model_seed{}.pt'.format('checkpoint', args.partition, args.seed))
        torch.save(discriminator.state_dict(), '{}/discriminator/seen{}_model_seed{}.pt'.format('checkpoint', args.partition, args.seed))
        # Generate counterfactual open set images
        
        f.write('end train.\n')
        f.write(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
        f.write('\n')
        '''
    with open(f'test_{args.seed}_{args.partition}.txt', 'a') as f:

        f.write('start test\n')
        f.write(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
        f.write('\n')
        encoder.load_state_dict(torch.load('{}/encoder/seen{}_model_seed{}.pt'.format('checkpoint', args.partition, args.seed), map_location=device))
        generator.load_state_dict(torch.load('{}/generator/seen{}_model_seed{}.pt'.format('checkpoint', args.partition, args.seed), map_location=device))
        discriminator.load_state_dict(torch.load('{}/discriminator/seen{}_model_seed{}.pt'.format('checkpoint', args.partition, args.seed), map_location=device))
        encoder.eval()
        generator.eval()
        classifier.eval()
        open_set_images = generate_counterfactuals(encoder, generator, classifier, load_training_dataset(args))
        import pdb
        pdb.set_trace()
        # Use counterfactual open set images to re-train the classifier
        augmented_classifier = classifier32(num_classes=6)
        augmented_classifier.load_state_dict(torch.load(f'/home/linhw/myproject/openmax/ckpt_deeper/classifier_epoch99_partition{partition}.pkl', map_location=device))
        
        trainset, testset = get_dataset('MNIST_id', download=False, seen=args.partition, train=False)
        _, oodset = get_dataset('MNIST_ood', download=False, seen=args.partition, train=False)
        trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=BATCH_SIZE, shuffle=True)
        testloader = torch.utils.data.DataLoader(dataset=testset, batch_size=BATCH_SIZE, shuffle=True)
        oodloader = torch.utils.data.DataLoader(dataset=oodset, batch_size=BATCH_SIZE, shuffle=True)

        augmented_classifier.train()
        for i in range(3):
            train_openset_classifier(trainloader, open_set_images, augmented_classifier)

        augmented_classifier.eval()
        correct = 0
        total = 0
        for image, label in testloader:
            pred = augmented_classifier(image.cuda()).max(1)[1]
            correct += (pred==label.cuda()).sum().data
            total += image.size(0)

        # Output ROC curves comparing the methods
        auc = test_open_set_performance(augmented_classifier, testloader, oodloader, mode='augmented_classifier')
        f.write(f'auc = {auc}')
        f.write('end test.\n')
        f.write(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
        f.write('\n')

def load_training_dataset(args):
    dataset, _ = get_dataset('MNIST_id', download=False, seen=args.partition, train=False)
    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True)
    def generator():
        for images, labels in dataloader:
            yield images.cuda(), labels.cuda()
    return generator()


def load_testing_dataset(args):
    _, testset = get_dataset('MNIST_id', download=False, seen=args.partition, train=False)
    dataloader = torch.utils.data.DataLoader(dataset=testset, batch_size=BATCH_SIZE, shuffle=True)
    def generator():
        for images, labels in dataloader:
            yield images.cuda(), labels.cuda()
    return generator()


def load_open_set(args):
    _, oodset = get_dataset('MNIST_ood', download=False, seen=args.partition, train=False)
    dataloader = torch.utils.data.DataLoader(dataset=oodset, batch_size=BATCH_SIZE, shuffle=True)
    def generator():
        for images, labels in dataloader:
            yield images.cuda(), labels.cuda()
    return generator()


def train_classifier(classifier, dataset):
    adam = torch.optim.Adam(classifier.parameters())
    for images, labels in dataset:
        adam.zero_grad()
        preds = F.log_softmax(classifier(images), dim=1)
        classifier_loss = F.nll_loss(preds, labels)
        classifier_loss.backward()
        adam.step()
        print('classifier loss: {}'.format(classifier_loss))


def test_classifier(classifier, dataset):
    total = 0
    total_correct = 0
    for images, labels in dataset:
        preds = classifier(images)
        correct = torch.sum(preds.max(dim=1)[1] == labels)
        total += len(images)
        total_correct += correct
    accuracy = float(total_correct) / total
    print('Test Accuracy: {}/{} ({:.03f})'.format(total_correct, total, accuracy))



def train_generative_model(encoder, generator, discriminator, dataset):
    generative_params = [x for x in encoder.parameters()] + [x for x in generator.parameters()]
    gen_adam = torch.optim.Adam(generative_params, lr=.005)
    disc_adam = torch.optim.Adam(discriminator.parameters(), lr=.02)
    for images, labels in dataset:
        disc_adam.zero_grad()
        fake = generator(torch.randn(len(images), LATENT_SIZE).cuda())
        disc_loss = torch.mean(F.softplus(discriminator(fake)) + F.softplus(-discriminator(images)))
        disc_loss.backward()
        gp_loss = calc_gradient_penalty(discriminator, images, fake)
        gp_loss.backward()
        disc_adam.step()

        gen_adam.zero_grad()
        mse_loss = torch.mean((generator(encoder(images)) - images) ** 2)
        mse_loss.backward()
        gen_loss = torch.mean(F.softplus(discriminator(images)))
        print('Autoencoder loss: {:.03f}, Generator loss: {:.03f}, Disc. loss: {:.03f}'.format(
            mse_loss, gen_loss, disc_loss))
        gen_adam.step()
    print('Generative training finished')


def calc_gradient_penalty(discriminator, real_data, fake_data, penalty_lambda=10.0):
    from torch import autograd
    alpha = torch.rand(real_data.size()[0], 1, 1, 1)
    alpha = alpha.expand(real_data.size())
    alpha = alpha.cuda()

    # Traditional WGAN-GP
    #interpolates = alpha * real_data + (1 - alpha) * fake_data
    # An alternative approach
    interpolates = torch.cat([real_data, fake_data])
    interpolates = interpolates.cuda()
    interpolates = autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = discriminator(interpolates)

    ones = torch.ones(disc_interpolates.size()).cuda()
    gradients = autograd.grad(
            outputs=disc_interpolates,
            inputs=interpolates,
            grad_outputs=ones,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]

    penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * penalty_lambda
    return penalty


def generate_counterfactuals(encoder, generator, classifier, dataset):
    cf_open_set_images = []
    idx = 0
    for images, labels in dataset:
        counterfactuals = generate_cf(encoder, generator, classifier, images)
        torchvision.utils.save_image(images.data.view(-1,1,28,28),
                       f'uca/{idx}_ori.png',
                       normalize=False,
                       nrow=12)
        torchvision.utils.save_image(counterfactuals.data.view(-1,1,28,28),
                       f'uca/{idx}_fake.png',
                       normalize=False,
                       nrow=12)
        cf_open_set_images.append(counterfactuals)
        idx += 1
        break
    print("Generated {} batches of counterfactual images".format(len(cf_open_set_images)))
    imutil.show(counterfactuals, filename='example_counterfactuals.jpg', img_padding=8)
    return cf_open_set_images


def generate_cf(encoder, generator, classifier, images,
                cf_iters=100, cf_step_size=.01, cf_distance_weight=1.0):
    from torch.autograd import grad

    # First encode the image into latent space (z)
    z_0 = encoder(images)
    z = z_0.clone()

    # Now perform gradient descent to update z
    for i in range(cf_iters):
        # Classify with one extra class
        logits = classifier(generator(z))
        augmented_logits = F.pad(logits, pad=(0,1))

        # Use the extra class as a counterfactual target
        batch_size, num_classes = logits.shape
        target_tensor = torch.LongTensor(batch_size).cuda()
        target_tensor[:] = num_classes

        # Maximize classification probability of the counterfactual target
        cf_loss = F.nll_loss(F.log_softmax(augmented_logits, dim=1), target_tensor)

        # Regularize with distance to original z
        distance_loss = torch.mean((z - z_0) ** 2)

        # Move z toward the "open set" class
        loss = cf_loss + distance_loss
        dc_dz = grad(loss, z, loss)[0]
        z -= cf_step_size * dc_dz

        # Sanity check: Clip gradients to avoid nan in ill-conditioned inputs
        #dc_dz = torch.clamp(dc_dz, -.1, .1)

        # Optional: Normalize to the unit sphere (match encoder's settings)
        z = norm(z)

    print("Generated batch of counterfactual images with cf_loss {:.03f}".format(cf_loss))
    recon = generator(z).detach()
    # Output the generated image as an example "unknown" image
    return generator(z).detach()


def test_open_set_performance(classifier, dataloader, oodloader, mode='confidence_threshold'):
    score = []
    for i, (images, labels) in enumerate(dataloader):
        images = images.cuda()
        logits = classifier(images)
        augmented_logits = F.pad(logits, pad=(0,1))
        # The implicit K+1th class (the open set class) is computed
        #  by assuming an extra linear output with constant value 0
        preds = F.softmax(augmented_logits)
        #preds = augmented_logits
        prob_unknown = preds[:, -1]
        prob_known = preds[:, :-1].max(dim=1)[0]
        prob_open = prob_unknown - prob_known

        score.extend(prob_open.data.cpu().numpy())

    openset_scores = []
    for i, (images, labels) in enumerate(oodloader):
        images = images.cuda()
        logits = classifier(images)
        augmented_logits = F.pad(logits, pad=(0,1))
        # The implicit K+1th class (the open set class) is computed
        #  by assuming an extra linear output with constant value 0
        preds = F.softmax(augmented_logits)
        #preds = augmented_logits
        prob_unknown = preds[:, -1]
        prob_known = preds[:, :-1].max(dim=1)[0]
        prob_open = prob_unknown - prob_known

        openset_scores.extend(prob_open.data.cpu().numpy())
    
    auc = plot_roc(score, openset_scores, mode)
    print('Detecting with mode {}, avg. known-class score: {}, avg unknown score: {}'.format(
        mode, np.mean(score), np.mean(openset_scores)))
    print('Mode {}: generated ROC with AUC score {:.03f}'.format(mode, auc))
    return auc


def get_score(preds, mode):
    if mode == 'confidence_threshold':
        return 1 - torch.max(torch.softmax(preds, dim=1), dim=1)[0].data.cpu().numpy()
    elif mode == 'augmented_classifier':
        return torch.softmax(preds, dim=1)[:, -1].data.cpu().numpy()
    assert False


def plot_roc(known_scores, unknown_scores, mode):
    from sklearn.metrics import roc_curve, roc_auc_score
    y_true = np.array([0] * len(known_scores) + [1] * len(unknown_scores))
    y_score = np.concatenate([known_scores, unknown_scores])
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    auc_score = roc_auc_score(y_true, y_score)
    title = 'ROC {}: AUC {:.03f}'.format(mode, auc_score)
    plot = plot_xy(fpr, tpr, x_axis="False Positive Rate", y_axis="True Positive Rate", title=title)
    filename = 'roc_{}.png'.format(mode)
    plot.figure.savefig(filename)
    return auc_score


def plot_xy(x, y, x_axis="X", y_axis="Y", title="Plot"):
    import pandas as pd
    # Hack to keep matplotlib from crashing when run without X
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # Apply sane defaults to matplotlib
    import seaborn as sns
    sns.set_style('darkgrid')

    # Generate plot
    df = pd.DataFrame({'x': x, 'y': y})
    plot = df.plot(x='x', y='y')
    plot.grid(b=True, which='major')
    plot.grid(b=True, which='minor')
    plot.set_title(title)
    plot.set_ylabel(y_axis)
    plot.set_xlabel(x_axis)
    return plot

if __name__ == '__main__':
    main()
