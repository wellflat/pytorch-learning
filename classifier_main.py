#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
from clearml import Task, Logger

from cifar10_loader import create_loaders
from resnet_trainer import ResNetTrainer

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('--train', '-t', action='store_true', help='training mode')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--epochs', default=200, type=int, help='epochs')
    parser.add_argument('--batch_size', default=256, type=int, help='batch size')
    parser.add_argument('--resume', '-r', action='store_true',
                        help='resume from checkpoint')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    print(args)
    task = Task.init(
        project_name='Image Classification',
        task_name='image_classification_CIFAR10',
        output_uri='./snapshot'
    )
    conf = {
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'base_lr': args.lr,
        'momentum': 0.9,
        'classes': 10
    }
    print(conf)
    conf = task.connect(conf)
    NET_PATH = './cifar_net.pth'
    loaders = create_loaders(conf)
    
    is_train = args.train
    logger = task.get_logger()
    trainer = ResNetTrainer(conf, logger, is_finetune=False)
    if is_train:
        trainer.train(loaders, conf['epochs'], resume=args.resume)
        trainer.save(NET_PATH)
    else:
        #trainer.load(NET_PATH)
        trainer.load_checkpoint()
        trainer.test(loaders['val'])
