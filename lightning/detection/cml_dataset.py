from typing import Dict, List, Tuple, Optional
import pickle
import numpy as np
import pandas as pd
from pandas import DataFrame
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from PIL import Image as PILImage
import albumentations as A
from allegroai import DataView
from config import DataConfig, Phase
from transform import get_transforms


CACHE_FILE = './cache/dataframe.pkl'

class DetectionDataset(Dataset):
    df: DataFrame
    image_ids: np.ndarray
    transform: A.Compose

    def __init__(self, df: DataFrame, transform: A.Compose=None) -> None:
        super().__init__()
        self.df = df
        self.image_ids = df['image_id'].unique()
        self.transform = transform


    def __getitem__(self, index: int) -> Tuple[torch.Tensor, Dict, str]:
        image_id = self.image_ids[index]
        row = self.df[self.df["image_id"] == image_id]
        image = PILImage.open(row['filepath'].values[0])
        image = np.array(image, dtype=np.float32)
        image /= 255.0
        boxes = row[["xmin", "ymin", "xmax", "ymax"]].values
        labels = row["class"].values

        sample = {
            "image": image,
            "bboxes": boxes,
            "labels": labels
        }
        if self.transform:
            sample = self.transform(**sample)
        
        image: Tensor = sample["image"]
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        area = torch.as_tensor(area, dtype=torch.float32)
        iscrowd = torch.zeros((row.shape[0], ), dtype=torch.int64)
        
        target = {
            "boxes": boxes,
            "labels": labels,
            "area": area,
            "iscrowd": iscrowd
        }

        return image, target, image_id


    def __len__(self):
        return self.image_ids.shape[0]


class DataModule(pl.LightningDataModule):
    config: DataConfig
    dataset: Dict[str, DetectionDataset]

    def __init__(self, config: DataConfig):
        super().__init__()
        self.config = config
        pl.seed_everything(self.config.random_state, workers=True)
    
    def prepare_data(self) -> None:
        return super().prepare_data()

    def setup(self, stage: Optional[str] = None) -> None:
        print(f'stage: {stage}')
        c = self.config
        df = get_clearml_dataset(
            c.dataset_name,
            c.version_name,
            c.query,
            use_cache=c.use_cache
        )
        train_df, val_df = self.__split_dataframe(df, c.train_fraction, c.random_state)

        train_dataset = DetectionDataset(train_df, get_transforms(phase=Phase.TRIAN))
        val_dataset = DetectionDataset(val_df, get_transforms(phase=Phase.VAL))

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


def get_clearml_dataset(dataset_name: str, version_name: str, query: str, use_cache: bool=True) -> DataFrame:
    if use_cache:
        print(f'use cache from {CACHE_FILE}')
        df = load_frame(CACHE_FILE)
        return df

    df = DataFrame(columns=["image_id", "class", "filepath", "width", "height", "xmin", "ymin", "xmax", "ymax"])
    dataview = DataView()
    dataview.add_query(
        dataset_name=dataset_name,
        version_name=version_name,
        roi_query=query
    )
    
    for frame in dataview.get_iterator():
        image_id = frame.source.split("/")[-1].split(".")[0]
        filepath = frame.get_local_source()
        annotation = frame.get_annotations()
        bboxes = [bbox.bounding_box_xywh for bbox in annotation]

        anno_list: List[Dict] = []
        for bbox in bboxes:
            pt = bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]
            row = {}
            row['xmin'] = bbox[0]
            row['ymin'] = bbox[1]
            row['xmax'] = bbox[0] + bbox[2]
            row['ymax'] = bbox[1] + bbox[3]
            row["width"] = bbox[2]
            row["height"] = bbox[3]
            row["image_id"] = image_id
            row["filepath"] = filepath
            #df = df.append(row, ignore_index=True)
            anno_list.append(row)
        df = pd.concat([df, pd.DataFrame(anno_list)], ignore_index=True)
    
    df = df.sort_values(by="image_id", ascending=True)
    df['class'] = 1  # 0: background
    print(f'save frame to {CACHE_FILE}')
    save_frame(df, CACHE_FILE)
    return df


def save_frame(df: DataFrame, filepath: str) -> None:
    with open(filepath, 'wb') as f:
        pickle.dump(df, f)


def load_frame(filepath: str) -> DataFrame:
    with open(filepath, 'rb') as f:
        df = pickle.load(f)
    return df


if __name__ == '__main__':
    c = DataConfig()
    data_module = DataModule(c)
    data_module.setup()
    for batch in data_module.train_dataloader():
        print(batch)