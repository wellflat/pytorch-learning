from dataclasses import dataclass


@dataclass
class DataConfig:
    train_filepath: str = '../../data/mnist/train.csv'
    test_filepath: str = '../../data/mnist/test.csv'
    batch_size: int = 100
    train_fraction = 0.8
    num_workers: int = 2


@dataclass
class TrainingConfig:
    lr: int = 1e-4
    step_size: int = 25
    gamma: float = 0.1
    num_classes: int = 10