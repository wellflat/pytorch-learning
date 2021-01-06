#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import sys
from clearml import Task

from oxfordpet_loader import create_loaders
from fasterrcnn_detector import FasterRCNNDetector

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('--train', '-t', action='store_true', help='training mode')
    parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
    parser.add_argument('--epochs', default=100, type=int, help='epochs')
    parser.add_argument('--batch_size', default=2, type=int, help='batch size')
    parser.add_argument('--resume', '-r', action='store_true',
                        help='resume from checkpoint')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    print(args)
    task_type = Task.TaskTypes.training if args.train else Task.TaskTypes.testing
    task = Task.init(
        project_name='Object Detection',
        task_name='object_detection_OXFORDPET',
        task_type=task_type,
        output_uri='./snapshot'
    )
    conf = {
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'base_lr': args.lr,
        'momentum': 0.9,
        'classes': 3
    }
    print(conf)
    conf = task.connect(conf)
    category = {'background':0, 'dog':1, 'cat':2}
    task.connect_label_enumeration(category)
    NET_PATH = './oxfordpet_net.pth'

    loaders = create_loaders(conf, use_cache=True)
    print(len(loaders['train']), len(loaders['val']))
    
    is_train = args.train
    estimator = FasterRCNNDetector(conf)
    #sys.exit(-1)
    if is_train:
        estimator.load_checkpoint(NET_PATH)
        estimator.fit(loaders, conf['epochs'], resume=args.resume)
        estimator.save(NET_PATH)
    else:
        #trainer.load(NET_PATH)
        estimator.load_checkpoint(NET_PATH)
        estimator.test(loaders['val'])
