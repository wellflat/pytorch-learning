from typing import Dict, Tuple, Optional
import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from config import DataConfig, Phase


class MNISTDataset(Dataset):
    n_pixels: int
    X: np.ndarray
    y: torch.Tensor
    transforms: transforms.Compose

    def __init__(self, df: pd.DataFrame, transform: transforms.Compose):
        super().__init__()
        self.n_pixels = 784
        self.transforms = transform
        
        if len(df.columns) == self.n_pixels:
            # test data
            # (batch_size, 784) -> (batch_size, 28, 28, 1)
            self.X = df.values.reshape((-1,28,28)).astype(np.uint8)[:,:,:,np.newaxis]
            self.y = None
        else:
            # training data
            # split label
            self.X = df.drop('label', axis=1).values.reshape((-1,28,28)).astype(np.uint8)[:,:,:,np.newaxis]
            #self.X = df.iloc[:,1:].values.reshape((-1,28,28)).astype(np.uint8)[:,:,:,np.newaxis]
            self.y = df['label'].to_numpy()
            #self.y = torch.from_numpy(df.iloc[:,0].values)
        
    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, index:int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.y is not None:
            return self.transforms(self.X[index]), self.y[index]
        else:
            return self.transforms(self.X[index])


class MNISTDataModule(pl.LightningDataModule):
    config: DataConfig
    dataset: Dict[Phase, Dataset]

    def __init__(self, config: DataConfig):
        super().__init__()
        self.config = config
    
    def setup(self, stage: Optional[str] = None) -> None:
        train = pd.read_csv(self.config.train_filepath) 
        test = pd.read_csv(self.config.test_filepath)
        train, val = self.__split_dataframe(train)
        transform = {
            'train': transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomAffine(degrees=45, translate=(0.1, 0.1), scale=(0.8, 1.2)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ]),
            'val': transforms.Compose([
                transforms.ToPILImage(),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])
        }
        self.dataset = {
            'train': MNISTDataset(train, transform['train']),
            'val': MNISTDataset(val, transform['val']),
            'test': MNISTDataset(test, transform['val']) 
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

    def __get_loader(self, phase: Phase) -> DataLoader:
        loader = DataLoader(
            self.dataset[phase],
            batch_size=self.config.batch_size,
            shuffle=True if phase == 'train' else False,
            num_workers=self.config.num_workers,
            pin_memory=True,
            drop_last=True if phase == "train" else False
        )
        return loader        

    def __split_dataframe(self, df: pd.DataFrame, fraction=0.9, state=1) -> Tuple[DataFrame, DataFrame]:
        df1 = df.sample(frac=fraction, random_state=state)
        df2 = df.drop(df1.index)
        return (df1, df2)
        


if __name__ == '__main__':
    train = pd.read_csv('../../data/mnist/train.csv') 
    #train_df = train.drop('label', axis=1)
    #print('train data:' + str(train_df.shape))
    transform = {
            'train': transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomAffine(degrees=45, translate=(0.1, 0.1), scale=(0.8, 1.2)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ]),
            'val': transforms.Compose([
                transforms.ToPILImage(),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])
    }
    ds = MNISTDataset(train, transform['train'])
    print(ds[0][0].shape)
