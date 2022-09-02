from typing import Any, Dict, List, Tuple, Optional
import math
import torch
from torch import nn, optim, Tensor
from torch.nn import functional as F
from torch.nn.parameter import Parameter
from torch.optim.lr_scheduler import StepLR, OneCycleLR
from torch.utils.tensorboard import SummaryWriter
import pytorch_lightning as pl
import torchmetrics
from torchvision import models
from torchvision.models.resnet import ResNet
from torchvision.transforms.functional import to_pil_image
from torchvision import transforms
import matplotlib.pyplot as plt

from config import TrainingConfig
from resnet import MNISTResNet


class Classifier(pl.LightningModule):
    net: nn.Module
    params: List[Parameter]
    criterion: nn.CrossEntropyLoss
    config: TrainingConfig

    def __init__(self, config: TrainingConfig):
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
        ## for debug image
        self.inv_trans = transforms.Compose([
            transforms.Normalize(mean = [ 0., 0., 0. ], std = [ 1/0.229, 1/0.224, 1/0.225 ]),
            transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ], std = [ 1., 1., 1. ]),
        ])

    def forward(self, x:Tensor) -> Tensor:
        return self.net(x)

    def configure_optimizers(self) -> Tuple[List[optim.Optimizer], List[optim.lr_scheduler.StepLR]]:
        #scheduler = StepLR(optimizer, step_size=self.config.step_size, gamma=self.config.gamma)
        optimizer = optim.AdamW(self.params, lr=self.config.base_lr)
        total_steps = self.trainer.estimated_stepping_batches
        print(f'total_steps: {total_steps}')
        scheduler = OneCycleLR(optimizer, max_lr=self.config.base_lr, total_steps=total_steps)
        scheduler = {"scheduler": scheduler, "interval" : "step" }  ## step for OneCycleLR
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
        ## debug image
        self.debug_image(inputs, probas, preds, batch_idx)
    
    def validation_epoch_end(self, outputs:Dict[str, Tensor]) -> None:
        return

    def test_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        logits: torch.Tensor = self.net(batch)
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

    def debug_image(self, inputs: torch.Tensor, probas: torch.Tensor, preds: torch.Tensor, batch_idx: int) -> None:
        writer: SummaryWriter = self.logger.experiment
        #input_id = batch_idx % len(inputs)
        input_id = random.randrange(len(inputs))
        inv_image = to_pil_image(self.inv_trans(inputs[input_id]))
        pred_class = preds[input_id].item()
        pred_proba = math.floor(probas[input_id][preds[input_id]].item()*1e+3) / 1e+3
        plt.imshow(inv_image)
        plt.title(f'{self.current_epoch}: {pred_class}')
        plt.xlabel(f"score: {pred_proba}")
        plt.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
        plt.show()