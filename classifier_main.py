#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
from clearml import Task
from cifar10_loader import create_loaders
from resnet_classifier import ResNetClassifier

def parse_args():
    parser = argparse.ArgumentParser(description='Classifier Training')
    parser.add_argument('--train', '-t', action='store_true', help='training mode')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--epochs', default=100, type=int, help='epochs')
    parser.add_argument('--batch_size', default=256, type=int, help='batch size')
    parser.add_argument('--resume', '-r', action='store_true',
                        help='resume from checkpoint')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    print(args)
    task = Task.init(
        project_name='Image Classification',
        task_name='image_classification_cifar10',
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
    estimator = ResNetClassifier(conf, logger, is_finetune=False)
    if is_train:
        estimator.fit(loaders, conf['epochs'], resume=args.resume)
        estimator.save(NET_PATH)
    else:
        estimator.load(NET_PATH)
        #estimator.test(loaders['val'])
        data = loaders['val'].__iter__()
        (inputs, targets) = data.next()
        print(inputs.shape, targets.shape)
        pred = estimator.predict(inputs)
        print(pred.max(1))
        