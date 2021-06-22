import os
import pdb
import numpy as np
from numpy.lib.twodim_base import triu_indices
import torch
from torch.utils.data.dataset import Subset
from torchvision import datasets, transforms

def set_random_seed(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    torch.cuda.manual_seed(seed_value)
    # torch.set_deterministic(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
DATA_PATH = '~/data/'

def get_transform(image_size=None):
    # Note: data augmentation is implemented in the layers
    # Hence, we only define the identity transformation here
    if image_size:  # use pre-specified image size
        train_transform = transforms.Compose([
            transforms.Resize((image_size[0], image_size[1])),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        test_transform = transforms.Compose([
            transforms.Resize((image_size[0], image_size[1])),
            transforms.ToTensor(),
        ])
    else:  # use default image size
        train_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        test_transform = transforms.ToTensor()

    return train_transform, test_transform

def MNIST_transform():
    train_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    return train_transform, test_transform

def get_dataset(dataset, download=False, seen=None, train=False):
    if 'MNIST' in dataset:
        train_transform, test_transform = MNIST_transform()
    else:
        train_transform, test_transform = get_transform()
    train_set = None
    test_set = None
    if dataset == 'cifar10':
        image_size = (32, 32, 3)
        n_classes = 10
        train_set = datasets.CIFAR10(DATA_PATH, train=True, download=download, transform=train_transform)
        test_set = datasets.CIFAR10(DATA_PATH, train=False, download=download, transform=test_transform)
    
    elif dataset == 'cifar100':
        image_size = (32, 32, 3)
        n_classes = 100
        train_set = datasets.CIFAR100(DATA_PATH, train=True, download=download, transform=train_transform)
        test_set = datasets.CIFAR100(DATA_PATH, train=False, download=download, transform=test_transform)

    elif dataset == 'svhn':
        test_set = datasets.SVHN(DATA_PATH, split='test', download=download, transform=test_transform)

    elif dataset == 'lsun_resize':
        test_dir = os.path.join(DATA_PATH, 'LSUN_resize')
        test_set = datasets.ImageFolder(test_dir, transform=test_transform)

    elif dataset == 'lsun_fix':
        test_dir = os.path.join(DATA_PATH, 'LSUN_fix')
        test_set = datasets.ImageFolder(test_dir, transform=test_transform)

    elif dataset == 'imagenet_resize':
        test_dir = os.path.join(DATA_PATH, 'Imagenet_resize')
        test_set = datasets.ImageFolder(test_dir, transform=test_transform)

    elif dataset == 'imagenet_fix':
        test_dir = os.path.join(DATA_PATH, 'Imagenet_fix')
        test_set = datasets.ImageFolder(test_dir, transform=test_transform)
    
  
    elif dataset == 'cifar10_ood':
        # dataset == 'cifar10_ood_1'
        image_size = (32, 32, 3)
        class_idx = [int(num) for num in seen] # seen 形如 '023456'
        for i in range(10):
            if i in class_idx:
                continue
            class_idx.append(i)
        test_set = datasets.CIFAR10(DATA_PATH, train=False, download=download, transform=test_transform, target_transform=lambda x:class_idx.index(x))
        test_class_idx = [6,7,8,9]
        test_set = my_get_subclass_dataset(test_set, test_class_idx)

    elif dataset == 'cifar10_id':
        # dataset == 'cifar10_ood_1'
        image_size = (32, 32, 3)
        n_classes = 6 # 6个seen类
        class_idx = [int(num) for num in seen] # seen 形如 '023456'
        for i in range(10):
            if i in class_idx:
                continue
            class_idx.append(i)
        train_set = datasets.CIFAR10(DATA_PATH, train=True, download=download, transform=train_transform, target_transform=lambda x:class_idx.index(x))
        test_set = datasets.CIFAR10(DATA_PATH, train=False, download=download, transform=test_transform, target_transform=lambda x:class_idx.index(x))
        test_class_idx = [0,1,2,3,4,5]
        train_set = my_get_subclass_dataset(train_set, test_class_idx)
        test_set = my_get_subclass_dataset(test_set, test_class_idx)

    elif dataset == 'cifar10_id_train': #用于计算score的threshold的train类
        # dataset == 'cifar10_ood_1'
        image_size = (32, 32, 3)
        n_classes = 6 # 6个seen类
        class_idx = [int(num) for num in seen] # seen 形如 '023456'
        for i in range(10):
            if i in class_idx:
                continue
            class_idx.append(i)
        train_set = datasets.CIFAR10(DATA_PATH, train=True, download=download, transform=train_transform, target_transform=lambda x:class_idx.index(x))
        test_set = datasets.CIFAR10(DATA_PATH, train=True, download=download, transform=test_transform, target_transform=lambda x:class_idx.index(x))
        test_class_idx = [0,1,2,3,4,5]
        train_set = my_get_subclass_dataset(train_set, test_class_idx)
        test_set = my_get_subclass_dataset(test_set, test_class_idx)
    
    elif dataset == 'MNIST_id_train': #用于计算score的threshold的train类
        # dataset == 'cifar10_ood_1'
        image_size = (32, 32, 3)
        n_classes = 6 # 6个seen类
        class_idx = [int(num) for num in seen] # seen 形如 '023456'
        for i in range(10):
            if i in class_idx:
                continue
            class_idx.append(i)
        train_set = datasets.MNIST(DATA_PATH, train=True, download=download, transform=train_transform, target_transform=lambda x:class_idx.index(x))
        test_set = datasets.MNIST(DATA_PATH, train=True, download=download, transform=test_transform, target_transform=lambda x:class_idx.index(x))
        test_class_idx = [0,1,2,3,4,5]
        train_set = my_get_subclass_dataset(train_set, test_class_idx)
        test_set = my_get_subclass_dataset(test_set, test_class_idx)

    elif dataset == 'svhn_id_train': #用于计算score的threshold的train类
        # dataset == 'cifar10_ood_1'
        image_size = (32, 32, 3)
        n_classes = 6 # 6个seen类
        class_idx = [int(num) for num in seen] # seen 形如 '023456'
        for i in range(10):
            if i in class_idx:
                continue
            class_idx.append(i)
        train_set = datasets.SVHN(DATA_PATH, split='train', download=download, transform=train_transform,target_transform=lambda x:class_idx.index(x))
        test_set = datasets.SVHN(DATA_PATH, split='train', download=download, transform=test_transform,target_transform=lambda x:class_idx.index(x))
        test_class_idx = [0,1,2,3,4,5]
        train_set = my_get_subclass_dataset(train_set, test_class_idx)
        test_set = my_get_subclass_dataset(test_set, test_class_idx)

    elif dataset == 'svhn_id':
        image_size = (32, 32, 3)
        n_classes = 6
        class_idx = [int(num) for num in seen] # seen 形如 '023456'
        for i in range(10):
            if i in class_idx:
                continue
            class_idx.append(i)
        train_set = datasets.SVHN(DATA_PATH, split='train', download=download, transform=train_transform,target_transform=lambda x:class_idx.index(x))
        test_set = datasets.SVHN(DATA_PATH, split='test', download=download, transform=test_transform,target_transform=lambda x:class_idx.index(x))
        test_class_idx = [0,1,2,3,4,5]
        train_set = my_get_subclass_dataset(train_set, test_class_idx)
        test_set = my_get_subclass_dataset(test_set, test_class_idx)

    elif dataset == 'svhn_ood':
        image_size = (32, 32, 3)
        class_idx = [int(num) for num in seen] # seen 形如 '023456'
        for i in range(10):
            if i in class_idx:
                continue
            class_idx.append(i)
        test_set = datasets.SVHN(DATA_PATH, split='test', download=download, transform=test_transform,target_transform=lambda x:class_idx.index(x))
        test_class_idx = [6,7,8,9]
        test_set = my_get_subclass_dataset(test_set, test_class_idx)
    
    elif dataset == 'MNIST_id':
        image_size = (32, 32, 3)
        n_classes = 6
        class_idx = [int(num) for num in seen] # seen 形如 '023456'
        for i in range(10):
            if i in class_idx:
                continue
            class_idx.append(i)
        train_set = datasets.MNIST(DATA_PATH, train=True, download=download, transform=train_transform,target_transform=lambda x:class_idx.index(x))
        test_set = datasets.MNIST(DATA_PATH, train=False, download=download, transform=test_transform,target_transform=lambda x:class_idx.index(x))
        test_class_idx = [0,1,2,3,4,5]
        train_set = my_get_subclass_dataset(train_set, test_class_idx)
        test_set = my_get_subclass_dataset(test_set, test_class_idx)
    
    elif dataset == 'MNIST_ood':
        # dataset == 'cifar10_ood_1'
        image_size = (32, 32, 3)
        class_idx = [int(num) for num in seen] # seen 形如 '023456'
        for i in range(10):
            if i in class_idx:
                continue
            class_idx.append(i)
        test_set = datasets.MNIST(DATA_PATH, train=False, download=download, transform=test_transform, target_transform=lambda x:class_idx.index(x))
        test_class_idx = [6,7,8,9]
        test_set = my_get_subclass_dataset(test_set, test_class_idx)

    elif dataset == 'cifar10_id+cifar10_ood':
        # ood 
        image_size = (32, 32, 3)
        n_classes = 6 # 6个seen类
        class_idx = [int(num) for num in seen] # seen 形如 '023456'
        train_set = datasets.CIFAR10(DATA_PATH, train=True, download=download, transform=train_transform, target_transform=lambda x:class_idx.index(x) if x in class_idx else n_classes)
        test_set = datasets.CIFAR10(DATA_PATH, train=False, download=download, transform=test_transform, target_transform=lambda x:class_idx.index(x) if x in class_idx else n_classes)
    
    elif dataset == 'MNIST_id+MNIST_ood':
        # ood 
        image_size = (32, 32, 3)
        n_classes = 6 # 6个seen类
        class_idx = [int(num) for num in seen] # seen 形如 '023456'
        train_set = datasets.MNIST(DATA_PATH, train=True, download=download, transform=train_transform, target_transform=lambda x:class_idx.index(x) if x in class_idx else n_classes)
        test_set = datasets.MNIST(DATA_PATH, train=False, download=download, transform=test_transform, target_transform=lambda x:class_idx.index(x) if x in class_idx else n_classes)

    elif dataset == 'svhn_id+svhn_ood':
        # ood 
        image_size = (32, 32, 3)
        n_classes = 6 # 6个seen类
        class_idx = [int(num) for num in seen] # seen 形如 '023456'
        train_set = datasets.SVHN(DATA_PATH, split='train', download=download, transform=test_transform,target_transform=lambda x:class_idx.index(x) if x in class_idx else n_classes)
        test_set = datasets.SVHN(DATA_PATH, split='test', download=download, transform=test_transform,target_transform=lambda x:class_idx.index(x) if x in class_idx else n_classes)

    elif dataset == 'cifar10+resizeLSUN':
        image_size = (32, 32, 3)
        n_classes = 10
        train_set = datasets.CIFAR10(DATA_PATH, train=True, download=download, transform=train_transform)
        test_cifar = datasets.CIFAR10(DATA_PATH, train=False, download=download, transform=train_transform)
        test_resizeLSUN = datasets.LSUN(root='/home/linhw/data', classes='test', 
            transform = transforms.Compose([
                    transforms.Resize([32,32]),
                    transforms.ToTensor()
            ]),
            target_transform=lambda x:10
        )
        print(len(test_cifar))
        test_LSUN = get_subset_with_len(test_resizeLSUN, len(test_cifar), shuffle=True)
        test_set = test_cifar + test_LSUN
    
    elif dataset == 'cifar10+cropLSUN':
        image_size = (32, 32, 3)
        n_classes = 10
        train_set = datasets.CIFAR10(DATA_PATH, train=True, download=download, transform=train_transform)
        test_cifar = datasets.CIFAR10(DATA_PATH, train=False, download=download, transform=train_transform)
        test_resizeLSUN = datasets.LSUN(root='/home/linhw/data', classes='test', 
            transform = transforms.Compose([
                    transforms.RandomCrop([32,32]),
                    transforms.ToTensor()
            ]),
            target_transform=lambda x:10
        )
        print(len(test_cifar))
        test_LSUN = get_subset_with_len(test_resizeLSUN, len(test_cifar), shuffle=True)
        test_set = test_cifar + test_LSUN
    
    elif dataset == 'cifar10+resizeImagenet':
        image_size = (32, 32, 3)
        n_classes = 10
        train_set = datasets.CIFAR10(DATA_PATH, train=True, download=download, transform=train_transform)
        test_cifar = datasets.CIFAR10(DATA_PATH, train=False, download=download, transform=train_transform)
        test_resizeimagenet = datasets.ImageFolder(root='/home/linhw/data/imagenet/large225', 
            transform = transforms.Compose([
                    transforms.Resize([32,32]),
                    transforms.ToTensor()
            ]),
            target_transform=lambda x:10
        )
        print(len(test_cifar))
        test_imagenet = get_subset_with_len(test_resizeimagenet, len(test_cifar), shuffle=True)
        test_set = test_cifar + test_imagenet

    elif dataset == 'cifar10+cropImagenet':
        image_size = (32, 32, 3)
        n_classes = 10
        train_set = datasets.CIFAR10(DATA_PATH, train=True, download=download, transform=train_transform)
        test_cifar = datasets.CIFAR10(DATA_PATH, train=False, download=download, transform=train_transform)
        test_cropimagenet = datasets.ImageFolder(root='/home/linhw/data/imagenet/large225',
            transform = transforms.Compose([
                    transforms.RandomCrop([32,32]),
                    transforms.ToTensor()
            ]),
            target_transform=lambda x:10
        )
        print(len(test_cifar))
        test_imagenet = get_subset_with_len(test_cropimagenet, len(test_cifar), shuffle=True)
        test_set = test_cifar + test_imagenet
    else:
        raise NotImplementedError()

    return train_set, test_set


def my_get_subclass_dataset(dataset, classes):
    if not isinstance(classes, list):
        classes = [classes]
    indices = []
    for idx, data in enumerate(dataset):
        if data[1] in classes:
            indices.append(idx)

    dataset = Subset(dataset, indices)
    return dataset

def get_subset_with_len(dataset, length, shuffle=False):
    set_random_seed(0)
    dataset_size = len(dataset)

    index = np.arange(dataset_size)
    if shuffle:
        np.random.shuffle(index)

    index = torch.from_numpy(index[0:length])
    subset = Subset(dataset, index)

    assert len(subset) == length

    return subset
