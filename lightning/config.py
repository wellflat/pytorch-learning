from dataclasses import dataclass


@dataclass
class DataConfig:
    train_filepath: str = '../train.csv'
    test_filepath: str = '../test.csv'
    batch_size: int = 100
    train_fraction = 0.8
    num_workers: int = 2


@dataclass
class TrainingConfig:
    lr: int = 0.01
    step_size: int = 25
    gamma: float = 0.1