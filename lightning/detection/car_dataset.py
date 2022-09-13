from typing import Dict, Tuple, Optional
from pathlib import Path
import numpy as np
import pandas as pd
from pandas import DataFrame
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from PIL import Image as PILImage
import albumentations as A
from torchvision.transforms.functional import to_pil_image
from config import DataConfig, Phase
from transform import get_transforms


class CarsDatasetAdaptor:
    image_dir: str
    df: pd.DataFrame

    def __init__(self, images_dir: str, df: pd.DataFrame):
        self.images_dir = Path(images_dir)
        self.df = df
        self.images = self.df['image'].unique().tolist()

    def __len__(self) -> int:
        return len(self.images)

    def get_image_and_labels_by_idx(self, index: int) -> Tuple[PILImage.Image, pd.DataFrame, np.ndarray, int, str]:
        image_name = self.images[index]
        image = PILImage.open(self.images_dir / image_name)
        pascal_bboxes = self.df[self.df['image'] == image_name][
            ["xmin", "ymin", "xmax", "ymax"]
        ].values
        class_labels = np.ones(len(pascal_bboxes))

        return image, pascal_bboxes, class_labels, index, image_name
    
    def show_image(self, index: int):
        image, bboxes, class_labels, image_id, image_name = self.get_image_and_labels_by_idx(index)
        print(f"image_id: {image_id}, name: {image_name}")
        #show_image(image, bboxes.tolist())
        #display_image(image, bboxes, class_labels, image_id)
        print(class_labels)


class DetectionDataset(Dataset):
    adaptor: CarsDatasetAdaptor
    transforms: A.Compose

    def __init__(self, adaptor: CarsDatasetAdaptor, transforms: A.Compose=None):
        super().__init__()
        self.adaptor = adaptor
        self.transforms = transforms
        
    def __getitem__(self, index) -> Tuple[torch.Tensor, Dict, str]:
        (image, boxes, labels, image_id, image_name) = self.adaptor.get_image_and_labels_by_idx(index)

        image = np.array(image, dtype=np.float32)
        image /= 255.0

        sample = {
            "image": image,
            "bboxes": boxes,
            "labels": labels,
        }
        if self.transforms:
            sample = self.transforms(**sample)
        
        image = sample["image"]

        boxes = np.array(sample["bboxes"])
        labels = sample["labels"]
        #tmp_image = to_pil_image(image)
        #print(f'image name: {image_name}')
        #display_image(tmp_image, boxes, labels, image_id)
        
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        area = torch.as_tensor(area, dtype=torch.float32)
        iscrowd = torch.zeros((len(labels), ), dtype=torch.int64)

        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([image_id], dtype=torch.int64),
            "area": area,
            "iscrowd": iscrowd
        }
        #target['boxes'] = torch.stack(tuple(map(torch.tensor, zip(*sample['bboxes'])))).permute(1, 0)
 
        return image, target, image_id
    
    def __len__(self):
        return len(self.adaptor)


class DataModule(pl.LightningDataModule):
    config: DataConfig
    dataset: Dict[str, DetectionDataset]

    def __init__(self, config: DataConfig):
        super().__init__()
        self.config = config
        pl.seed_everything(36, workers=True)
    
    def prepare_data(self) -> None:
        return super().prepare_data()

    def setup(self, stage: Optional[str] = None) -> None:
        print(f'stage: {stage}')
        #df = parse_xmls(f'{self.config.xml_dir}/*.xml')
        df = pd.read_csv(self.config.train_filepath)
        train_df, val_df = self.__split_dataframe(df, self.config.train_fraction, self.config.random_state)
        train_adaptor = CarsDatasetAdaptor(self.config.image_dir, train_df)
        val_adaptor = CarsDatasetAdaptor(self.config.image_dir, val_df)
        train_dataset = DetectionDataset(train_adaptor, get_transforms(phase=Phase.TRIAN))
        val_dataset = DetectionDataset(val_adaptor, get_transforms(phase=Phase.VAL))

        self.dataset = {
            'train': train_dataset,
            'val': val_dataset,
            'test': val_dataset
        }
        print(f"train: {len(self.dataset['train'])}, val: {len(self.dataset['val'])}")
    
    def train_dataloader(self) -> DataLoader:
        return self.__get_loader('train')

    def val_dataloader(self) -> DataLoader:
        return self.__get_loader('val')

    def test_dataloader(self) -> DataLoader:
        return self.__get_loader('test')
    
    def predict_dataloader(self) -> DataLoader:
        return self.__get_loader('test')

    def __get_loader(self, phase: str) -> DataLoader:
        loader = DataLoader(
            self.dataset[phase],
            batch_size=self.config.batch_size,
            shuffle=True if phase == 'train' else False,
            collate_fn=self.__collate_fn,
            num_workers=self.config.num_workers,
            pin_memory=True,
            drop_last=True if phase == "train" else False
        )
        return loader

    def __collate_fn(self, batch):
        return tuple(zip(*batch))
    
    def __split_dataframe(self, df: DataFrame, fraction: float, state: int=1) -> Tuple[DataFrame, DataFrame]:
        df1 = df.sample(frac=fraction, random_state=state)
        df2 = df.drop(df1.index)
        return (df1, df2)

def save_df(df: pd.DataFrame, filepath: str):
    with open(filepath, 'wb') as f:
        pickle.dump(df, f)

def load_df(filepath: str) -> pd.DataFrame:
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    return data


if __name__ == '__main__':
    train_path = '../../data/car/training_images'
    df = pd.read_csv('../../data/car/train_solution_bounding_boxes.csv')
    print(df.head())
    car_ds = CarsDatasetAdaptor(train_path, df)
    car_ds.show_image(123)
