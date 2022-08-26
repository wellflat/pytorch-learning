from typing import Tuple
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms


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
