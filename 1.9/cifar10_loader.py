from typing import Dict
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms


def create_loaders(conf: Dict) -> Dict[str, DataLoader]:
    print('==> loading cifar10 dataset')
    transform = {
        'train': transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]),
        'val': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    }
    '''
    transform = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }'''
    
    trainset = torchvision.datasets.CIFAR10(
        root='./data',
        train=True,
        download=True,
        transform=transform['train']
    )
    trainloader = DataLoader(
        trainset, batch_size=conf['batch_size'],
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    testset = torchvision.datasets.CIFAR10(
        root='./data',
        train=False,
        download=True,
        transform=transform['val']
    )
    testloader = DataLoader(
        testset, batch_size=200,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    print(trainset.data.shape)
    print(testset.data.shape)
    print(trainset.classes)

    return {
        'train': trainloader,
        'val': testloader
    }