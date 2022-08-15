from typing import Dict, Tuple
from glob import glob
import pickle
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
import torch
from torch.utils.data import Dataset
from PIL import Image
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from plot import display_image

class xml2list(object):
    
    def __init__(self, classes):
        self.classes = classes
        
    def __call__(self, xml_path):
        
        ret = []
        
        xml = ET.parse(xml_path).getroot()
        
        for size in xml.iter("size"):
          
            width = float(size.find("width").text)
            height = float(size.find("height").text)
                
        for obj in xml.iter("object"):
            
            difficult = int(obj.find("difficult").text)
            if difficult == 1:
                continue
                
            bndbox = [width, height]
            
            name = obj.find("name").text.lower().strip() 
            bbox = obj.find("bndbox") 
            
            pts = ["xmin", "ymin", "xmax", "ymax"]
            
            for pt in pts:
                
                cur_pixel =  float(bbox.find(pt).text)
                    
                bndbox.append(cur_pixel)
                
            label_idx = self.classes.index(name)
            bndbox.append(label_idx)
            
            ret += [bndbox]
            
        return np.array(ret) # [width, height, xmin, ymin, xamx, ymax, label_idx]

def parse_xmls(xml_path: str='./data/oxford/annotations/xmls/*.xml') -> pd.DataFrame:
    print('parsing annotation data start.')
    xml_paths = glob(xml_path)
    classes = ["dog", "cat"]

    transform_anno = xml2list(classes)

    df = pd.DataFrame(columns=["image_id", "width", "height", "xmin", "ymin", "xmax", "ymax", "class"])

    for path in xml_paths:
        image_id = path.split("/")[-1].split(".")[0]
        bboxs = transform_anno(path)
    
        for bbox in bboxs:
            tmp = pd.Series(bbox, index=["width", "height", "xmin", "ymin", "xmax", "ymax", "class"])
            tmp["image_id"] = image_id
            df = pd.concat([df, pd.DataFrame([tmp])], ignore_index=True)
            #df = df.append(tmp, ignore_index=True)

    df = df.sort_values(by="image_id", ascending=True)
    df['class'] = df['class'] + 1
    #print(df.head())
    print('parsing annotation data complete.')
    return df


class DetectionDataset(Dataset):
    image_ids: np.ndarray
    df: pd.DataFrame
    image_dir: str
    transforms: A.Compose

    def __init__(self, df: pd.DataFrame, image_dir: str):
        super().__init__()
        self.image_ids = df["image_id"].unique()
        self.df = df
        self.image_dir = image_dir
        self.transforms = A.Compose([
            A.Resize(height=512, width=512, p=1),
            A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ToTensorV2(p=1),
        ],
        p=1.0,
        bbox_params=A.BboxParams(
            format="pascal_voc", min_area=0, min_visibility=0, label_fields=["labels"]
        ))
        
    def __getitem__(self, index) -> Tuple[torch.Tensor, Dict, str]:
        image_id: str = self.image_ids[index]
        image = Image.open(f"{self.image_dir}/{image_id}.jpg")
        
        
        records = self.df[self.df["image_id"] == image_id]
        boxes = torch.tensor(records[["xmin", "ymin", "xmax", "ymax"]].values, dtype=torch.float32)

        width: int = records["width"].values[0]
        height: int = records["height"].values[0]
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        area = torch.as_tensor(area, dtype=torch.float32)
        
        labels = torch.tensor(records["class"].values, dtype=torch.int32)
        
        iscrowd = torch.zeros((records.shape[0], ), dtype=torch.int32)

        #display_image(image, boxes, labels, image_id)

        sample = {
            'image': np.array(image, dtype=np.float32),
            'bboxes': boxes,
            'labels': labels
        }
        sample = self.transforms(**sample)
        sample["bboxes"] = np.array(sample["bboxes"])
        sample["bboxes"][:, [0, 1, 2, 3]] = sample["bboxes"][
            :, [1, 0, 3, 2]
        ]  # convert to yxyx
        image = sample['image']
        boxes = sample['bboxes']
        labels = sample['labels']
        
        target = {}
        #target["boxes"] = torch.as_tensor(boxes, dtype=torch.float32)
        target["labels"]= torch.as_tensor(labels),
        target["image_id"] = torch.tensor([index])
        #target["area"] = area
        #target["img_size"] = torch.tensor([width, height], dtype=torch.int32)
        #target["img_scale"] = torch.tensor([1.])
        #target["iscrowd"] = iscrowd

        ## for EfficientDet
        target["bbox"] = torch.as_tensor(boxes, dtype=torch.float32)
        target["cls"] = target["labels"]
        
        #{
        #  'labels': [tensor([[1],[2]], dtype=torch.int32)],
        #  'image_id': tensor([[3366],[ 853]]),
        #  'bbox': tensor([[[204.8000, 1.5375, 367.6160, 330.5706]], [[140.2880, 124.6492, 290.8160, 325.6964]]]),
        #  'cls': [tensor([[1],[2]], dtype=torch.int32)]
        #}
        #({'labels': (tensor([1], dtype=torch.int32),), 'image_id': tensor([3366]), 'bbox': tensor([[204.8000,   1.5375, 367.6160, 330.5706]]), 'cls': (tensor([1], dtype=torch.int32),)},
        # {'labels': (tensor([2], dtype=torch.int32),), 'image_id': tensor([853]), 'bbox': tensor([[140.2880, 124.6492, 290.8160, 325.6964]]), 'cls': (tensor([2], dtype=torch.int32),)})
        # {'bbox': [tensor([[  1.5375, 204.8000, 330.5706, 367.6160]]), tensor([[124.6492, 140.2880, 325.6964, 290.8160]])], 'cls': [(tensor([1], dtype=torch.int32),), (tensor([2], dtype=torch.int32),)]}

        return image, target, image_id
    
    def __len__(self):
        return self.image_ids.shape[0]


def save_df(df: pd.DataFrame, filepath: str):
    with open(filepath, 'wb') as f:
        pickle.dump(df, f)

def load_df(filepath: str) -> pd.DataFrame:
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    return data
