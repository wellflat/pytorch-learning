from dataclasses import dataclass


@dataclass
class DataConfig:
    image_dir: str = '/content/data/training_images'
    #xml_dir: str = '../../data/oxford/annotations/xmls'
    train_filepath: str = '/content/data/train_solution_bounding_boxes (1).csv'
    use_cache: bool = True
    batch_size: int = 8
    train_fraction = 0.8
    num_workers: int = 2


@dataclass
class TrainingConfig:
    #base_lr: float = 1e-2 # 0.01 for SGD
    base_lr: float = 1e-4  # 0.0001 for AdamW
    step_size: int = 25
    gamma: float = 0.1
    img_size: int = 512
    num_classes: int = 2
    conf_threshold: float = 0.2
    nms_threshold: float = 0.3
