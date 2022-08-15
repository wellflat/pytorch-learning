from typing import Dict, Tuple, Optional
import pandas as pd
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from config import DataConfig
from car_dataset import DetectionDataset, CarsDatasetAdaptor

#Phase = Literal['train', 'val', 'test']

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
        #df = parse_xmls(f'{self.config.xml_dir}/*.xml')
        df = pd.read_csv(self.config.train_filepath)
        adaptor = CarsDatasetAdaptor(self.config.image_dir, df)
        dataset = DetectionDataset(adaptor)
        print(f'len: {len(dataset)}')
        n_train = int(len(dataset) * self.config.train_fraction)
        n_val = len(dataset) - n_train
        train, val = random_split(dataset, [n_train, n_val])
        
        self.dataset = {
            'train': train,
            'val': val,
            'test': val
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
            collate_fn=self.collate_fn,
            num_workers=self.config.num_workers,
            pin_memory=True,
            drop_last=True# if phase == "train" else False
        )
        return loader

    def collate_fn(self, batch):
        return tuple(zip(*batch))


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

    