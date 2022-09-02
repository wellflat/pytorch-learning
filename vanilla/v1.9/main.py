#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
from cifar10_loader import create_loaders
from classifier import Classifier
from PIL import Image

def parse_args():
    parser = argparse.ArgumentParser(description='Classifier Training')
    parser.add_argument('--train', '-t', action='store_true', default=False, help='training mode')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--epochs', default=100, type=int, help='epochs')
    parser.add_argument('--batch_size', default=200, type=int, help='batch size')
    parser.add_argument('--resume', '-r', action='store_true',
                        help='resume from checkpoint')
    return parser.parse_args()

def load_image(filepath):
    image = Image.open(filepath).convert('RGB')
    return image


if __name__ == '__main__':
    args = parse_args()
    print(args)
    
    conf = {
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'base_lr': args.lr,
        'momentum': 0.9,
        'num_classes': 10
    }
    print(conf)
    NET_PATH = './cifar10_net.pth'
    loaders = create_loaders(conf)
    is_train = args.train
    estimator = Classifier(conf)
    if is_train:
        estimator.fit(loaders, conf['epochs'], resume=args.resume)
        estimator.save(NET_PATH)
    else:
        imagepath = './image/airplane4.png'
        image = load_image(imagepath)
        estimator.load(NET_PATH)
        #estimator.test(loaders['val'])
        label, confidence = estimator.predict(image)
        print(label, confidence)
        '''
        data = loaders['val'].__iter__()
        (inputs, targets) = data.next()
        print(inputs.shape, targets.shape)
        pred = estimator.predict(inputs)
        print(pred.max(1))'''
        
        