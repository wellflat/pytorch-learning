from typing import Any, Dict, List, Tuple, Optional
import torch
from torch import nn, optim, Tensor
from torch.nn import functional as F
from torch.nn.parameter import Parameter
from torch.optim.lr_scheduler import _LRScheduler, StepLR, OneCycleLR
import pytorch_lightning as pl
from torchvision import models
from torchvision.models.resnet import ResNet
import torchmetrics
from config import ModelConfig
from resnet import MNISTResNet
from debug_image import display_image


class Classifier(pl.LightningModule):
    net: nn.Module
    params: List[Parameter]
    criterion: nn.CrossEntropyLoss
    config: ModelConfig

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        #self.net = MNISTResNet()
        self.net = self.create_model(config.num_classes)
        self.params = [p for p in self.net.parameters() if p.requires_grad]
        self.criterion = nn.CrossEntropyLoss()
        average = 'macro' if config.num_classes == 2 else 'micro'
        self.metrics = torchmetrics.MetricCollection([
            torchmetrics.Accuracy(num_classes=config.num_classes, average=average),
            torchmetrics.Precision(num_classes=config.num_classes, average=average),
            torchmetrics.Recall(num_classes=config.num_classes, average=average),
            torchmetrics.F1Score(num_classes=config.num_classes, average=average)
        ])
        self.save_hyperparameters()
        self.on_debug_image = config.on_debug_image

    def forward(self, x:Tensor) -> Tensor:
        return self.net(x)

    def configure_optimizers(self) -> Tuple[List[optim.Optimizer], List[_LRScheduler]]:
        #scheduler = StepLR(optimizer, step_size=self.config.step_size, gamma=self.config.gamma)
        optimizer = optim.AdamW(self.params, lr=self.config.base_lr)
        total_steps = self.trainer.estimated_stepping_batches
        print(f'total_steps: {total_steps}')
        lr = OneCycleLR(optimizer, max_lr=self.config.base_lr, total_steps=total_steps)
        scheduler = {"scheduler": lr, "interval" : "step"}  ## step for OneCycleLR
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
        probas = F.softmax(logits, dim=1)
        preds = probas.argmax(dim=1)
        #accuracy = sum(preds == targets) / len(targets)
        self.metrics(preds, targets)
        self.log_dict(self.metrics, prog_bar=True, on_step=False, on_epoch=True)

        if self.on_debug_image:
            display_image(
                self.logger.experiment,
                self.current_epoch,
                inputs,
                probas,
                preds,
                batch_idx
            )
    
    def validation_epoch_end(self, outputs:Dict[str, Tensor]) -> None:
        return

    def test_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        logits: Tensor = self.net(batch)
        preds = logits.argmax(dim=1)
        return preds

    def test_epoch_end(self, outputs: Tensor) -> Tensor:
        preds = torch.cat(outputs)
        return preds
    
    def predict_step(self, batch: Tensor, batch_idx:int) -> Tensor:
        logits: Tensor = self.net(batch)
        probas = F.softmax(logits, dim=1)
        preds = probas.argmax(dim=1)
        return preds

    def create_model(self, num_classes: int) -> ResNet:
        net = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        for p in net.parameters():
            p.requires_grad = False
        '''
        net.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=64,
            kernel_size=net.conv1.kernel_size,
            stride=net.conv1.stride,
            padding=net.conv1.padding,
            bias=False
        )'''
        fc_input_dim = net.fc.in_features
        net.fc = nn.Linear(fc_input_dim, num_classes)
        #print(net)
        return net

    