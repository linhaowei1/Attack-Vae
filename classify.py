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
def train(model, optimizer, dataloader, device):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()
    for batch_idx, (inputs, targets) in tqdm(enumerate(dataloader), total=len(dataloader), desc="training"):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    train_acc = correct/total
    running_loss = train_loss/(batch_idx+1)

    return train_acc, running_loss

def test(model, dataloader, device):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    batch_num = 0 
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in tqdm(enumerate(dataloader), total=len(dataloader), desc="testing"):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            batch_num += 1 
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    test_acc = correct/total
    running_loss = test_loss/len(dataloader)
    return test_acc,running_loss

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
    if args.dataset == 'MNIST':
        transform=transforms.Compose([
            transforms.ToTensor()
        ])
        transform_train = transform
        tramsform_test = transform
    else:
        raise NotImplementedError
    
    trainset = torchvision.datasets.MNIST(DATA_PATH, train=True, transform=transform)
    if args.mode == 'ood_train':
        trainset = my_get_subclass_dataset(trainset, [0,1,2,3,4,5])
    elif args.mode == 'adv_ood_train':
        trainset = my_get_subclass_dataset(trainset, [0,1,2,3,4,5,6,7,8,9])
        advset = torchvision.datasets.ImageFolder(
            '/home/linhw/myproject/Attack-Vae/untargeted_generate',
            transform=transforms.Compose([
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor()
            ]),
            target_transform=lambda x: 10
        )
        trainset += advset
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=8)

    testset = torchvision.datasets.MNIST(DATA_PATH, train=False, transform=transform)
    if args.mode == 'ood_train' or args.mode == 'adv_ood_train':
        testset = my_get_subclass_dataset(testset, [0,1,2,3,4,5,6,7,8,9])
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=8)
    epochs = args.epochs
    if args.mode == 'test':
        GEN_PATH = f'/home/linhw/myproject/Attack-Vae/gen_imgs-{epochs}'
        genset = ImageFolder(GEN_PATH, transform=transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor()]
        ))
        genloader = torch.utils.data.DataLoader(genset, batch_size=batch_size, shuffle=False, num_workers=8)
    # Model
    logger.info('==> Building model..')
    if args.model == 'smallCNN':
        model = SmallCNN()
        if args.mode == 'ood_train':
            model.classifier[-1] = nn.Linear(200, 6)
        elif args.mode == 'adv_ood_train':
            model.classifier[-1] = nn.Linear(200, 11)
    model = model.to(device)
    if args.mode == 'test':
        model.load_state_dict(torch.load(args.params, map_location=device)['state_dict'])

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    if args.dataset == 'MNIST':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.7)
    
    best_acc = 0
    if args.mode == 'train' or args.mode == 'ood_train' or args.mode == 'adv_ood_train':
        for epoch in range(args.epochs):
            logger.info("Epoch {} started".format(epoch))

            train_acc,training_loss = train(model, optimizer, trainloader, device)
            logger.info("train acc = {:.4f}, training loss = {:.4f}".format(train_acc, training_loss))

            test_acc, test_loss = test(model, testloader, device)
            logger.info("test acc = {:.4f}, test loss = {:.4f}".format(test_acc, test_loss))

            if test_acc > best_acc:
                best_acc = test_acc
                logger.info("best acc improved to {:.4f}".format(best_acc))
                model_to_save = model.module if hasattr(model, 'module') else model
                torch.save(model_to_save.state_dict(), '{}/{}model.pt'.format(output_dir, args.seen))
                logger.info("model saved to {}/model.pt".format(output_dir))

            scheduler.step()

            logger.info("Epoch {} ended, best acc = {:.4f}".format(epoch, best_acc))
        
        logger.info("Training finished, best_acc = {:.4f}".format(best_acc))
    
    if args.mode == 'test':

        test_acc, test_loss = test(model, testloader, device)
        logger.info("test acc = {:.4f}, test loss = {:.4f}".format(test_acc, test_loss))

        gen_acc, gen_loss = test(model, genloader, device)
        logger.info("gen acc = {:.4f}, gen loss = {:.4f}".format(gen_acc, gen_loss))

        logger.info("test finished")


if __name__ == '__main__':
    main()
