from typing import Any, Dict, List, Tuple, Optional
import torch
from torch import nn, optim, Tensor
from torch.optim.lr_scheduler import StepLR
import pytorch_lightning as pl
from resnet import MNISTResNet
from config import TrainingConfig


class Classifier(pl.LightningModule):
    net: nn.Module
    criterion: nn.CrossEntropyLoss
    config: TrainingConfig

    def __init__(self, config:TrainingConfig):
        super().__init__()
        self.config = config
        self.net = MNISTResNet()
        self.criterion = nn.CrossEntropyLoss()
        self.save_hyperparameters()

    def forward(self, x:Tensor) -> Tensor:
        return self.net.forward(x)

    def configure_optimizers(self) -> Tuple[List[optim.Optimizer], List[optim.lr_scheduler.StepLR]]:
        optimizer = optim.Adam(self.parameters(), lr=self.config.lr)
        scheduler = StepLR(optimizer, step_size=self.config.step_size, gamma=self.config.gamma)
        return [optimizer], [scheduler]
    
    def training_step(self, batch:Tuple[Tensor, Tensor], batch_idx:Optional[int]) -> Dict[str, Tensor]:
        inputs, targets = batch
        logits: Tensor = self.net(inputs)
        loss: Tensor = self.criterion(logits, targets)
        return {'loss': loss}
    
    def training_epoch_end(self, outputs:Dict[str, Tensor]) -> None:
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.log('train_loss', avg_loss, prog_bar=True)
    
    def validation_step(self, batch:Tuple[Tensor, Tensor] , batch_idx:int) -> Dict[str, float]:
        inputs, targets = batch
        logits: Tensor = self.net(inputs)
        preds = logits.argmax(1)
        accuracy = sum(preds == targets) / len(targets)
        return {'val_acc': accuracy}
    
    def validation_epoch_end(self, outputs:Dict[str, Tensor]) -> None:
        avg_acc = torch.stack([x['val_acc'] for x in outputs]).mean()
        self.log('val_acc', avg_acc, prog_bar=True)

    def test_step(self, batch:Tensor, batch_idx:int) -> Tensor:
        logits: torch.Tensor = self.net(batch)
        preds = logits.argmax(1)
        return preds

    def test_epoch_end(self, outputs:Tensor) -> Tensor:
        preds = torch.cat(outputs)
        return preds
    
    def predict_step(self, batch, batch_idx:int) -> Tensor:
        logits: Tensor = self.net(batch)
        preds = logits.argmax(1)
        return preds