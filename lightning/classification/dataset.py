from typing import Dict, Optional
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset, random_split
from torchvision.datasets import ImageFolder
import pytorch_lightning as pl
import albumentations as A
from config import DataConfig, Phase
from transform import get_transforms


class DatasetFromSubset(Dataset):
    subset: Subset
    transform: A.Compose

    def __init__(self, subset: Subset, transform: A.Compose=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index: int):
        pil_image, label = self.subset[index]
        np_image = np.array(pil_image)
        if self.transform:
            image = self.transform(image=np_image)['image']

        return image, label

    def __len__(self):
        return len(self.subset)


class DataModule(pl.LightningDataModule):
    config: DataConfig
    dataset: Dict[str, Dataset]

    def __init__(self, config: DataConfig):
        super().__init__()
        self.config = config
        pl.seed_everything(36, workers=True)
    
    def prepare_data(self) -> None:
        super().prepare_data()

    def setup(self, stage: Optional[str] = None) -> None:
        print(f'stage: {stage}')
        dataset = ImageFolder(self.config.image_dir)
        n_train = int(len(dataset) * self.config.train_fraction)
        n_val = len(dataset) - n_train
        train_subset, val_subset = random_split(dataset, [n_train, n_val])
        train_dataset = DatasetFromSubset(train_subset, transform=get_transforms(Phase.TRIAN))
        val_dataset = DatasetFromSubset(val_subset, transform=get_transforms(Phase.VAL))
        
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
            #collate_fn=self.__collate_fn,
            num_workers=self.config.num_workers,
            pin_memory=True,
            drop_last=True if phase == "train" else False
        )
        return loader