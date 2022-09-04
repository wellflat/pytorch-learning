from typing import Dict, Optional
from torch.utils.data import DataLoader
from torchvision import transforms
import pytorch_lightning as pl
from config import DataConfig, Phase
import torchvision



class CIFAR10DataModule(pl.LightningDataModule):
    config: DataConfig
    dataset: Dict[Phase, DataLoader]

    def __init__(self, config: DataConfig):
        super().__init__()
        self.config = config
    
    def setup(self, stage: Optional[str] = None) -> None:
        transform = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }
        trainset = torchvision.datasets.CIFAR10(
            root='./data',
            train=True,
            download=True,
            transform=transform['train']
        )
        valset = torchvision.datasets.CIFAR10(
            root='./data',
            train=False,
            download=True,
            transform=transform['val']
        )
        self.dataset = {
            'train': trainset,
            'val': valset
        }
    
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


if __name__ == '__main__' :
    config = DataConfig()
    #data_module = MNISTDataModule(config)
    #data_module.setup('fit')
    data_module = CIFAR10DataModule(config)
    data_module.setup('fit')