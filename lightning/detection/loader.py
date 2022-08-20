from typing import Dict, Tuple, Optional
import pandas as pd
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from config import DataConfig, Phase
from car_dataset import DetectionDataset, CarsDatasetAdaptor
from transform import get_transforms

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
        train_transforms = get_transforms(phase=Phase.TRIAN)
        val_transforms = get_transforms(phase=Phase.VAL)
        train_dataset = DetectionDataset(train_adaptor, train_transforms)
        val_dataset = DetectionDataset(val_adaptor, val_transforms)
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
    
    def __split_dataframe(self, df: pd.DataFrame, fraction: float, state: int=1) -> Tuple[pd.DataFrame, pd.DataFrame]:
        df1 = df.sample(frac=fraction, random_state=state)
        df2 = df.drop(df1.index)
        return (df1, df2)


if __name__ == '__main__':
    data_config = DataConfig()
    data_module = DataModule(data_config)
    data_module.setup()
    loader = data_module.train_dataloader()
    c = 0
    for sample in loader:
        print(sample)
        c += 1
        if c > 3:
            break

    