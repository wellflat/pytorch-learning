from typing import Dict, List, Tuple, Optional
import torch
from torch import optim, Tensor
from torch.optim.lr_scheduler import _LRScheduler, StepLR, OneCycleLR
import pytorch_lightning as pl
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn, FasterRCNN_MobileNet_V3_Large_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from config import TrainingConfig


class Detector(pl.LightningModule):
    net: FasterRCNN
    config: TrainingConfig
    metrics: MeanAveragePrecision

    def __init__(self, config: TrainingConfig):
        super().__init__()
        self.config = config
        self.net = self.create_model(num_claases=config.num_classes)
        self.params = [p for p in self.net.parameters() if p.requires_grad]
        self.metrics = MeanAveragePrecision()
        self.save_hyperparameters()

    def forward(self, images: Tensor, targets: Dict=None) -> Tensor:
        if targets is not None:
            outputs = self.net(images, targets)
        else:
            outputs = self.net(images)
        return outputs

    def configure_optimizers(self) -> Tuple[List[optim.Optimizer], List[_LRScheduler]]:
        optimizer = optim.AdamW(self.params, lr=self.config.base_lr)
        #optimizer = optim.SGD(self.params, lr=self.config.base_lr, momentum=0.9, weight_decay=5e-4)
        #scheduler = StepLR(optimizer, step_size=self.config.step_size, gamma=self.config.gamma)
        total_steps = self.trainer.estimated_stepping_batches
        print(f'total_steps: {total_steps}')
        scheduler = OneCycleLR(optimizer, max_lr=self.config.base_lr, total_steps=total_steps)
        scheduler = {"scheduler": scheduler, "interval" : "step" }  ## step for OneCycleLR
        return [optimizer], [scheduler]
    
    def training_step(self, batch:Tuple[Tensor, Dict, str], batch_idx:int) -> Dict[str, Tensor]:
        inputs, targets, ids = batch
        loss_dict: Dict = self.net(inputs, targets)
        loss: torch.Tensor = sum(loss for loss in loss_dict.values())
        #self.log("train_loss", loss, on_step=True, on_epoch=False, prog_bar=True, batch_size=1)
        return {'loss': loss}
    
    def training_epoch_end(self, outputs:Dict[str, Tensor]) -> None:
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.log('train_loss', avg_loss, prog_bar=True)
    
    def validation_step(self, batch: Tuple[Tensor, Dict, str] , batch_idx: int) -> None:
        inputs, targets, ids = batch
        preds: List[Dict] = self.net(inputs)
        self.metrics.update(preds, targets)
    
    def validation_epoch_end(self, outputs: Dict[str, Tensor]) -> None:
        map = self.metrics.compute()
        self.log_dict(map, prog_bar=True)
        self.metrics.reset()

    def predict_step(self, input: Tensor, batch_idx: int) -> Tensor:
        preds = self.net(input)
        return preds

    def create_model(self, num_claases: int=2, pretrained: bool=True) -> FasterRCNN:
        ## documentation
        ## https://pytorch.org/vision/master/models/generated/torchvision.models.detection.fasterrcnn_resnet50_fpn.html#torchvision.models.detection.fasterrcnn_resnet50_fpn
        '''
        net = fasterrcnn_resnet50_fpn_v2(
            weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT,
            trainable_backbone_layers=3
        )
        in_features = net.roi_heads.box_predictor.cls_score.in_features
        net.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_claases)
        '''
        '''
        RoIHeads(
          (box_roi_pool): MultiScaleRoIAlign()
          (box_head): TwoMLPHead(
            (fc6): Linear(in_features=12544, out_features=1024, bias=True)
            (fc7): Linear(in_features=1024, out_features=1024, bias=True)
          )
          (box_predictor): FastRCNNPredictor(
            (cls_score): Linear(in_features=1024, out_features=2, bias=True)
            (bbox_pred): Linear(in_features=1024, out_features=8, bias=True)
          )
        )'''
        
        net = fasterrcnn_mobilenet_v3_large_fpn(
            weights=FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT,
            num_claases=num_claases,
            trainable_backbone_layers=3
        )
        in_features = net.roi_heads.box_predictor.cls_score.in_features
        net.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_claases)
        #print(net.roi_heads)
        
        return net