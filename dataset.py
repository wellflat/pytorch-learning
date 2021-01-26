#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Dict, Tuple
import pickle
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.transforms import Compose, ToTensor
from PIL import Image
from allegroai import DataView


class DetectionDataset(Dataset):
    df: pd.DataFrame
    image_ids: np.ndarray
    image_dir: str
    transform: Compose


    def __init__(self, df: pd.DataFrame, transform: Compose=None) -> None:
        super().__init__()
        self.df = df
        self.image_ids = df['image_id'].unique()
        if transform is None:
            self.transform = Compose([ToTensor()])
        else:
            self.transform = transform


    def __getitem__(self, index: int) -> Tuple[torch.Tensor, Dict, str]:
        image_id = self.image_ids[index]
        records = self.df[self.df["image_id"] == image_id]
        image = Image.open(records['filepath'].values[0])
        image = self.transform(image)
        boxes = torch.tensor(records[["xmin", "ymin", "xmax", "ymax"]].values, dtype=torch.float32)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        area = torch.as_tensor(area, dtype=torch.float32)
        labels = torch.tensor(records["class"].values, dtype=torch.int64)
        iscrowd = torch.zeros((records.shape[0], ), dtype=torch.int64)
        
        target = {}
        target["image_id"] = torch.tensor([index])
        target["labels"]= labels
        target["boxes"] = boxes
        target["area"] = area
        target["iscrowd"] = iscrowd

        return image, target, image_id


    def __len__(self):
        return self.image_ids.shape[0]



def get_allegro_dataset(dataset_name:str, version_name: str, query: str) -> pd.DataFrame:
    df = pd.DataFrame(columns=["image_id", "class", "filepath", "width", "height", "xmin", "ymin", "xmax", "ymax"])

    dataview = DataView()
    dataview.add_query(
        dataset_name=dataset_name,
        version_name=version_name,
        roi_query=query
    )
    
    for frame in dataview.get_iterator():
        #print(frame)
        image_id = frame.source.split("/")[-1].split(".")[0]
        filepath = frame.get_local_source()
        annotation = frame.get_annotations()
        bboxes = [bbox.bounding_box_xywh for bbox in annotation[1:]]

        for bbox in bboxes:
            pt = bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]
            row = pd.Series(pt, index=["xmin", "ymin", "xmax", "ymax"])
            row["width"] = bbox[2]
            row["height"] = bbox[3]
            row["image_id"] = image_id
            row["filepath"] = filepath
            df = df.append(row, ignore_index=True)
    
    df = df.sort_values(by="image_id", ascending=True)
    df['class'] = 1  # 0: background
    return df


def create_loaders(dataset_name: str, version_name: str, query: str) -> Dict[str, DataLoader]:
    df = get_allegro_dataset(dataset_name, version_name, query)
    dataset = DetectionDataset(df)
    n_train = int(len(dataset) * 0.8)
    n_val = len(dataset) - n_train
    generator = torch.manual_seed(2021)
    train, val = random_split(dataset, [n_train, n_val], generator)

    def collate_fn(batch):
        return tuple(zip(*batch))

    trainloader = DataLoader(
        train,
        batch_size=2,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True
    )
    valloader = DataLoader(
        val,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True
    )

    return {
        'train': trainloader,
        'val': valloader
    }

if __name__ == '__main__':
    dataset_name = 'Data registration example2'
    #version_name = 'NSFW,Jan'
    version_name = 'Gun,Jan'
    query = 'SM'
    loaders = create_loaders(dataset_name, version_name, query)
    print(loaders['train'])

    for i, (inputs, targets, image_ids) in enumerate(loaders['train']):
        print(image_ids)
    

