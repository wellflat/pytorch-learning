#!/usr/bin/env python

from typing import List
import argparse
import pandas as pd
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback, DeviceStatsMonitor, LearningRateMonitor, ModelCheckpoint, EarlyStopping, RichProgressBar
from pytorch_lightning.loggers import TensorBoardLogger
from loader import DataModule
from detector import Detector
from config import DataConfig, ModelConfig
from clearml import Task, TaskTypes


def parse_args():
    parser = argparse.ArgumentParser(description='Detector Training')
    parser.add_argument('--train', '-t', action='store_true', default=False, help='training mode')
    parser.add_argument('--batch_size', '-b', default=8, type=int, help='batch size')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--epoch', default=100, type=int, help='epochs')
    parser.add_argument('--gpu', action='store_true', default=False, help='use gpu')
    parser.add_argument('--offline', action='store_true', default=False, help='offline mode')
    return parser.parse_args()


def setup_clearml() -> Task:
    task = Task.init(
        project_name='Test',
        task_name='car-detection',
        task_type=TaskTypes.training,
        tags=None,
        reuse_last_task_id=True,
        continue_last_task=False,
        output_uri=None,
        auto_connect_arg_parser=True,
        auto_connect_frameworks=True,
        auto_resource_monitoring=True,
        auto_connect_streams=True,
    )
    task.connect_label_enumeration({'background': 0, 'car': 1})
    return task


def get_trainer_callbacks() -> List[Callback]:
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
        DeviceStatsMonitor(),
        RichProgressBar(refresh_rate=1)
    ]
    return trainer_callbacks

if __name__ == '__main__':
    args = parse_args()
    print(args)
    if not args.offline:
        task = setup_clearml()
    data_config = DataConfig(batch_size=args.batch_size)
    train_config = ModelConfig(base_lr=args.lr)
    data_module = DataModule(data_config)
    model = Detector(train_config)
    logger = TensorBoardLogger('tb_logs', name='Car - FasterRCNN')
    trainer_callbacks = get_trainer_callbacks()
    trainer = pl.Trainer(
        max_epochs=args.epoch,
        callbacks=trainer_callbacks,
        logger=logger,
        log_every_n_steps=20,
        num_sanity_val_steps=1,
        accelerator='gpu' if args.gpu else 'cpu',
        benchmark=True,
        precision=16 if args.gpu else 32,
        amp_backend='native'
    )
    model_path = 'detector.ckpt'
    if args.train:
        trainer.fit(model, data_module)
        print(f'best model: {trainer_callbacks[0].best_model_path}')
        trainer.save_checkpoint(model_path)
        if not args.offline:
            Task.close(task)
    else:
        print(f'load from {model_path}')
        model = model.load_from_checkpoint(model_path, config=train_config)
        trainer.validate(model, data_module)
        #trainer.test(model, data_module)
        #preds = trainer.predict(model, data_module)
        #result = torch.cat(preds)  