from typing import Any, Dict, List, Tuple, Optional
import torch
from torch import nn, optim, Tensor
from torch.optim.lr_scheduler import StepLR
import pytorch_lightning as pl
import torchmetrics
from torchvision import models
from torchvision.models.resnet import ResNet
from config import TrainingConfig
from resnet import MNISTResNet

class Classifier(pl.LightningModule):
    net: nn.Module
    criterion: nn.CrossEntropyLoss
    config: TrainingConfig

    def __init__(self, config: TrainingConfig):
        super().__init__()
        self.config = config
        #self.net = MNISTResNet()
        self.net = self.create_model(config.num_classes)
        self.params = [p for p in self.net.parameters() if p.requires_grad]
        self.criterion = nn.CrossEntropyLoss()
        self.save_hyperparameters()
        self.metrics = torchmetrics.MetricCollection([
            torchmetrics.Accuracy(),
            torchmetrics.Precision(),
            torchmetrics.Recall(),
            torchmetrics.F1Score()
        ])
        self.save_hyperparameters()

    def forward(self, x:Tensor) -> Tensor:
        return self.net.forward(x)

    def configure_optimizers(self) -> Tuple[List[optim.Optimizer], List[optim.lr_scheduler.StepLR]]:
        optimizer = optim.Adam(self.params, lr=self.config.lr)
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
    
    def validation_step(self, batch:Tuple[Tensor, Tensor] , batch_idx:int) -> None:
        inputs, targets = batch
        logits: Tensor = self.net(inputs)
        preds = logits.argmax(1)
        #accuracy = sum(preds == targets) / len(targets)
        self.metrics(preds, targets)
        self.log_dict(self.metrics, prog_bar=True, on_step=False, on_epoch=True)
    
    def validation_epoch_end(self, outputs:Dict[str, Tensor]) -> None:
        return

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

    def create_model(self, num_classes: int) -> ResNet:
        net = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        for p in net.parameters():
            p.requires_grad = False
        net.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=64,
            kernel_size=net.conv1.kernel_size,
            stride=net.conv1.stride,
            padding=net.conv1.padding,
            bias=False
        )
        fc_input_dim = net.fc.in_features
        net.fc = nn.Linear(fc_input_dim, num_classes)
        #print(net)
        return net