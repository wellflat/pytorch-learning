from dataclasses import dataclass
from enum import Enum

@dataclass
class DataConfig:
    image_dir: str = '../../data/car/training_images'
    #xml_dir: str = '../../data/oxford/annotations/xmls'
    train_filepath: str = '../../data/car/train_solution_bounding_boxes.csv'
    dataset_name: str = 'modelgun'
    version_name: str = 'Current'
    query: str = 'SM'
    use_cache: bool = False
    batch_size: int = 2
    train_fraction: float = 0.8
    random_state: int = 36
    num_workers: int = 2


@dataclass
class ModelConfig:
    arch: str = 'mobilenetv3' #'resnet50' 
    num_classes: int = 2
    base_lr: int = 1e-4  ## AdamW: 1e-4, SGD: 1e-2
    step_size: int = 25  ## for StepLR
    gamma: float = 0.1
    conf_threshold: float = 0.2
    nms_threshold: float = 0.3
    on_debug_image: bool = True  ## for MLOps tools

class Phase(Enum):
    TRIAN = 'train'
    VAL = 'val'
    TEST = 'test'
