#!/usr/bin/env python
# -*- coding: utf-8 -*-


from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
from clearml import Task

from trainer import Trainer


if __name__ == '__main__':
    task = Task.init(
        project_name='Image Classification',
        task_name='image_classification_CIFAR10'
    )
    conf = {
        'epochs': 100,
        'batch_size': 256,
        'base_lr': 0.1,
        'momentum': 0.9,
        'classes': 10
    }
    conf = task.connect(conf)

    print('==> Preparing data..')
    transforms = {
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
    transforms = {
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
        transform=transforms['train']
    )
    trainloader = DataLoader(
        trainset, batch_size=conf['batch_size'],
        shuffle=True, num_workers=2
    )

    testset = torchvision.datasets.CIFAR10(
        root='./data',
        train=False,
        download=True,
        transform=transforms['val']
    )
    testloader = DataLoader(
        testset, batch_size=200,
        shuffle=False, num_workers=2
    )

    print(trainset.data.shape)
    print(testset.data.shape)
    print(trainset.classes)

    print(len(testloader.dataset))
    classes = trainset.classes

    NET_PATH = './cifar_net.pth'
    loaders = {
        'train': trainloader,
        'val': testloader
    }
    is_train = True
    trainer = Trainer(conf, is_finetune=False)
    if is_train:
        trainer.train(loaders, conf['epochs'], resume=False)
        trainer.save(NET_PATH)
    else:
        trainer.load(NET_PATH)
        trainer.test(testloader)
