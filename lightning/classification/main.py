#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Dict, List
import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback, DeviceStatsMonitor, EarlyStopping, LearningRateMonitor, ModelCheckpoint, RichProgressBar
from pytorch_lightning.loggers import TensorBoardLogger
from dataset import DataModule
from classifier import Classifier
from config import DataConfig, ModelConfig
from clearml import Task, TaskTypes


def parse_args():
    parser = argparse.ArgumentParser(description='Classifier Training')
    parser.add_argument('--train', '-t', action='store_true', default=False, help='training mode')
    parser.add_argument('--batch_size', '-b', default=100, type=int, help='batch size')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--epoch', default=100, type=int, help='epochs')
    parser.add_argument('--gpu', action='store_true', default=False, help='use gpu')
    parser.add_argument('--offline', action='store_true', default=False, help='offline mode')
    return parser.parse_args()


def setup_clearml(project_name: str, task_name: str, labels: Dict[str, int]=None) -> Task:
    task = Task.init(
        project_name=project_name,
        task_name=task_name,
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
    if labels is not None:
        task.connect_label_enumeration(labels)
    return task


def get_trainer_callbacks(task_name: str) -> List[Callback]:
    model_filename = task_name + ':{epoch:02d}-{Accuracy:.3f}'
    trainer_callbacks = [
        ModelCheckpoint(
            dirpath='./checkpoints',
            filename=model_filename,
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
        DeviceStatsMonitor(),
        RichProgressBar(refresh_rate=1)
    ]
    return trainer_callbacks


def get_nowstr() -> str:
    import datetime
    delta = datetime.timedelta(hours=9)
    JST = datetime.timezone(delta, 'JST')
    now = datetime.datetime.now(JST)
    return f'{now:%m%d%H%M}'


if __name__ == '__main__':
    args = parse_args()
    print(args)
    task_name = f'classification-{get_nowstr()}'
    if not args.offline:
        task = setup_clearml('Test', task_name)

    data_config = DataConfig(batch_size=args.batch_size)
    data_module = DataModule(data_config)
    model_config = ModelConfig(base_lr=args.lr)
    model = Classifier(model_config)
    logger = TensorBoardLogger('tb_logs', name='Classifier Training')
    trainer_callbacks = get_trainer_callbacks()
    trainer = pl.Trainer(
        max_epochs=args.epoch,
        callbacks=trainer_callbacks,
        logger=logger,
        log_every_n_steps=50,
        num_sanity_val_steps=1,
        accelerator='gpu' if args.gpu else 'cpu',
        benchmark=True,
        precision=16 if args.gpu else 32, #'bf16',
        amp_backend="native"
    )
    model_path = 'classifier.ckpt'
    if args.train:
        trainer.fit(model, data_module)
        print(f'best model: {trainer_callbacks[0].best_model_path}')
        trainer.save_checkpoint(model_path)
        if not args.offline:
            Task.close(task)
    else:
        print(f'load from {model_path}')
        model = model.load_from_checkpoint(model_path, config=model_config)
        trainer.validate(model, data_module)
        #trainer.test(model, data_module)
        #preds = trainer.predict(model, data_module)
        #result = torch.cat(preds)