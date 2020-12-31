

from typing import Dict
from glob import glob
import pickle
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from PIL import Image


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
            df = df.append(tmp, ignore_index=True)

    df = df.sort_values(by="image_id", ascending=True)
    #print(df.head())
    print('parsing annotation data complete.')
    return df


class DetectionDataset(Dataset):
    
    def __init__(self, df: pd.DataFrame, image_dir: str):
        
        super().__init__()
        
        self.image_ids = df["image_id"].unique()
        self.df = df
        self.image_dir = image_dir
        
    def __getitem__(self, index):

        transform = transforms.Compose([transforms.ToTensor()])

        # 入力画像の読み込み
        image_id = self.image_ids[index]
        image = Image.open(f"{self.image_dir}/{image_id}.jpg")
        image = transform(image)
        
        # アノテーションデータの読み込み
        records = self.df[self.df["image_id"] == image_id]
        boxes = torch.tensor(records[["xmin", "ymin", "xmax", "ymax"]].values, dtype=torch.float32)
        
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        area = torch.as_tensor(area, dtype=torch.float32)
        
        labels = torch.tensor(records["class"].values, dtype=torch.int64)
        
        iscrowd = torch.zeros((records.shape[0], ), dtype=torch.int64)
        
        target = {}
        target["boxes"] = boxes
        target["labels"]= labels
        target["image_id"] = torch.tensor([index])
        target["area"] = area
        target["iscrowd"] = iscrowd
        
        return image, target, image_id
    
    def __len__(self):
        return self.image_ids.shape[0]


def create_loaders(conf: Dict, use_cache: bool=True) -> Dict[str, DataLoader]:
    image_dir = './data/oxford/images'
    cache_filepath = './data/oxford/oxfordpet_dataset.pkl'
    if use_cache:
        dataset = load(cache_filepath)
    else:
        df = parse_xmls()
        dataset = DetectionDataset(df, image_dir)
        save(dataset, cache_filepath)
    
    torch.manual_seed(2021)
    n_train = int(len(dataset) * 0.7)
    n_val = len(dataset) - n_train
    train, val = random_split(dataset, [n_train, n_val])
    def collate_fn(batch):
        return tuple(zip(*batch))

    trainloader = DataLoader(train, batch_size=conf.get('batch_size', 1), shuffle=True, collate_fn=collate_fn)
    valloader = DataLoader(val, batch_size=2, shuffle=False, collate_fn=collate_fn)
    return {
        'train': trainloader,
        'val': valloader
    }

def save(dataset: DetectionDataset, filepath: str):
    with open(filepath, 'wb') as f:
        pickle.dump(dataset, f)

def load(filepath: str) -> DetectionDataset:
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    return data


if __name__ == '__main__':
    conf = {'batch_size': 2}
    loaders = create_loaders()
    print(len(loaders['train']), len(loaders['val']))