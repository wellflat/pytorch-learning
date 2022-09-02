#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import sys
import pandas as pd
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import DeviceStatsMonitor, LearningRateMonitor, ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from loader import MNISTDataModule
from classifier import Classifier
from config import DataConfig, TrainingConfig


def parse_args():
    parser = argparse.ArgumentParser(description='Classifier Training')
    parser.add_argument('--train', '-t', action='store_true', default=False, help='training mode')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--epoch', default=100, type=int, help='epochs')
    parser.add_argument('--batch_size', '-b', default=100, type=int, help='batch size')
    parser.add_argument('--offline', action='store_true', default=False, help='offline mode')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    print(args)
    data_config = DataConfig(batch_size=args.batch_size)
    train_config = TrainingConfig(args.lr)
    data_module = MNISTDataModule(data_config)
    model = Classifier(train_config)
    logger = TensorBoardLogger('tb_logs', name='Classifier Training')
    trainer_callbacks = [
        ModelCheckpoint(
            dirpath='./checkpoints',
            filename='model:{epoch:02d}-{val_acc:.3f}',
            monitor='Accuracy',
            mode='max',
            save_top_k=1
        ),
        EarlyStopping(
            monitor='Accuracy',
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
        log_every_n_steps=50,
        num_sanity_val_steps=1,
        accelerator='gpu',
        benchmark=True,
        precision=16,
        amp_backend="native"
    )
    model_path = 'mnist-lenet.ckpt'
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