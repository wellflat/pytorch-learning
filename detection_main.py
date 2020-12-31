#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import sys
from clearml import Task

from oxfordpet_loader import create_loaders
from fasterrcnn_trainer import FasterRCNNTrainer

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('--train', '-t', action='store_true', help='training mode')
    parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
    parser.add_argument('--epochs', default=100, type=int, help='epochs')
    parser.add_argument('--batch_size', default=1, type=int, help='batch size')
    parser.add_argument('--resume', '-r', action='store_true',
                        help='resume from checkpoint')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    print(args)
    task = Task.init(
        project_name='Object Detection',
        task_name='object_detection_OXFORDPET',
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
    NET_PATH = './oxfordpet_net.pth'

    loaders = create_loaders(conf, use_cache=False)
    #print(len(loaders['train']), len(loaders['val']))
    
    is_train = args.train
    trainer = FasterRCNNTrainer(conf)
    if is_train:
        trainer.train(loaders, conf['epochs'], resume=args.resume)
        trainer.save(NET_PATH)
    else:
        #trainer.load(NET_PATH)
        trainer.load_checkpoint()
        trainer.test(loaders['val'])
