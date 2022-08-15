from typing import Dict, Tuple
from pathlib import Path
import numpy as np
import pandas as pd
import pickle
import torch
from torch.utils.data import Dataset
from PIL import Image as PILImage
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from torchvision.transforms.functional import to_pil_image


class CarsDatasetAdaptor:
    def __init__(self, images_dir_path: str, annotations_df: pd.DataFrame):
        self.images_dir_path = Path(images_dir_path)
        self.annotations_df = annotations_df
        self.images = self.annotations_df.image.unique().tolist()

    def __len__(self) -> int:
        return len(self.images)

    def get_image_and_labels_by_idx(self, index: int) -> Tuple[PILImage.Image, pd.DataFrame, np.ndarray, int, str]:
        image_name = self.images[index]
        image = PILImage.open(self.images_dir_path / image_name)
        pascal_bboxes = self.annotations_df[self.annotations_df.image == image_name][
            ["xmin", "ymin", "xmax", "ymax"]
        ].values
        class_labels = np.ones(len(pascal_bboxes))

        return image, pascal_bboxes, class_labels, index, image_name
    
    def show_image(self, index):
        image, bboxes, class_labels, image_id, image_name = self.get_image_and_labels_by_idx(index)
        print(f"image_id: {image_id}, name: {image_name}")
        #show_image(image, bboxes.tolist())
        display_image(image, bboxes, class_labels, image_id)
        print(class_labels)


class DetectionDataset(Dataset):
    ds: CarsDatasetAdaptor
    transforms: A.Compose

    def __init__(self, adaptor: CarsDatasetAdaptor):
        super().__init__()
        self.adaptor = adaptor
        
        self.transforms = A.Compose([
            A.Resize(height=512, width=512, p=1),
            #A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  ## default mean and std
            #A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), 
            ToTensorV2(p=1)
        ],
        p=1.0,
        bbox_params=A.BboxParams(
            format="pascal_voc", min_area=0, min_visibility=0, label_fields=["labels"]
        ))

        ###self.transforms = transforms.Compose([transforms.ToTensor()])
        
    def __getitem__(self, index) -> Tuple[torch.Tensor, Dict, str]:
        (image, boxes, labels, image_id, image_name) = self.adaptor.get_image_and_labels_by_idx(index)

        image = np.array(image, dtype=np.float32)
        image /= 255.0

        sample = {
            "image": image,
            "bboxes": boxes,
            "labels": labels,
        }
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
