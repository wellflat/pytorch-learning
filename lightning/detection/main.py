#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import sys
import pandas as pd
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import DeviceStatsMonitor, LearningRateMonitor, ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from loader import DataModule
from detector import Detector
from config import DataConfig, TrainingConfig


def parse_args():
    parser = argparse.ArgumentParser(description='Detection Training')
    parser.add_argument('--train', '-t', action='store_true', default=False, help='training mode')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--epoch', default=100, type=int, help='epoch')
    parser.add_argument('--batch_size', default=10, type=int, help='batch size')
    parser.add_argument('--use_gpu', action='store_true', default=False, help='use gpu')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    print(args)
    data_config = DataConfig(batch_size=args.batch_size)
    train_config = TrainingConfig(base_lr=args.lr)
    data_module = DataModule(data_config)
    model = Detector(train_config)
    logger = TensorBoardLogger('tb_logs', name='Car - FasterRCNN')
    trainer_callbacks = [
        ModelCheckpoint(
            dirpath='./checkpoints',
            filename='car-detection:{epoch:02d}-{map:.3f}',
            monitor='map',
            mode='max',
            save_top_k=1
        ),
        EarlyStopping(
            monitor='map',
            mode='max',
            patience=10
        ),
        LearningRateMonitor('epoch'),
        DeviceStatsMonitor()
    ]
    trainer = pl.Trainer(
        max_epochs=args.epoch,
        callbacks=trainer_callbacks,
        logger=logger,
        log_every_n_steps=20,
        num_sanity_val_steps=1,
        accelerator='gpu' if args.use_gpu else 'cpu',
        precision=16,
        amp_backend='native'
    )
    model_path = 'fasterrcnn-detector.ckpt'
    if args.train:
        trainer.fit(model, data_module)
        print(f'best model: {trainer_callbacks[0].best_model_path}')
        trainer.save_checkpoint(model_path)
    else:
        print(f'load from {model_path}')
        model = model.load_from_checkpoint(model_path, config=train_config)
        #trainer.validate(model, data_module)
        #trainer.test(model, data_module)
        preds = trainer.predict(model, data_module)
        result = torch.cat(preds)
        submission = pd.read_csv('../submission/sample_submission.csv')
        submission['Label'] = result
        submission.to_csv('submission.csv', index=False)   