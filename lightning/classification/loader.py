from typing import Dict, Tuple, Literal, Optional
import pandas as pd
from pandas import DataFrame
from torch.utils.data import DataLoader
from torchvision import transforms
import pytorch_lightning as pl
from config import DataConfig
from dataset import MNISTDataset

Phase = Literal['train', 'val', 'test']

class MNISTDataModule(pl.LightningDataModule):
    config: DataConfig
    dataset: Dict[Phase, DataLoader]

    def __init__(self, config:DataConfig):
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

    def __split_dataframe(self, df:DataFrame, fraction=0.9, state=1) -> Tuple[DataFrame, DataFrame]:
        df1 = df.sample(frac=fraction, random_state=state)
        df2 = df.drop(df1.index)
        return (df1, df2)


if __name__ == '__main__' :
    config = DataConfig()
    data_module = MNISTDataModule(config)
    data_module.setup('fit')