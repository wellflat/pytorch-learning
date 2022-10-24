from typing import Dict, List, Tuple
import torch
from torch import nn, optim, Tensor
from torch.nn.parameter import Parameter
from torch.optim.lr_scheduler import _LRScheduler, OneCycleLR
import pytorch_lightning as pl
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.models import detection
from torchvision.ops import batched_nms
from config import ModelConfig
from debug_image import display_image


Batch = Tuple[Tuple[Tensor], Tuple[Dict[str, Tensor]], Tuple[int]]

class Detector(pl.LightningModule):
    net: nn.Module
    params: List[Parameter]
    config: ModelConfig
    metrics: MeanAveragePrecision

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.net = self.create_model(num_classes=config.num_classes, arch=config.arch)
        self.params = [p for p in self.net.parameters() if p.requires_grad]
        self.metrics = MeanAveragePrecision()
        self.save_hyperparameters()
        self.on_debug_image = config.on_debug_image

    def forward(self, images: Tensor, targets: Dict=None) -> Tensor:
        if targets is not None:
            outputs = self.net(images, targets)
        else:
            outputs = self.net(images)
        return outputs

    def configure_optimizers(self) -> Tuple[List[optim.Optimizer], List[_LRScheduler]]:
        total_steps = self.trainer.estimated_stepping_batches
        print(f'total_steps: {total_steps}')
        if self.config.optimizer == 'radam':
            optimizer = optim.RAdam(self.params, lr=self.config.base_lr)
        else:
            optimizer = optim.SGD(self.params, lr=self.config.base_lr, momentum=0.9, weight_decay=5e-4)
        scheduler = OneCycleLR(optimizer, max_lr=self.config.base_lr, total_steps=total_steps)
        scheduler = {'scheduler': scheduler, 'interval' : 'step'}  ## step for OneCycleLR
        return [optimizer], [scheduler]
    
    def training_step(self, batch: Batch, batch_idx: int) -> Dict[str, Tensor]:
        inputs, targets, ids = batch
        loss_dict: Dict = self.net(inputs, targets)
        loss: torch.Tensor = sum(loss for loss in loss_dict.values())
        #self.log("train_loss", loss, on_step=True, on_epoch=False, prog_bar=True, batch_size=1)
        return {'loss': loss}
    
    def training_epoch_end(self, outputs: Dict[str, Tensor]) -> None:
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.log('train_loss', avg_loss, prog_bar=True)
    
    def validation_step(self, batch: Batch , batch_idx: int) -> None:
        inputs, targets, ids = batch
        preds: List[Dict] = self.net(inputs)
        self.metrics.update(preds, targets)

        if self.on_debug_image:
            display_image(
                self.logger.experiment,
                self.current_epoch,
                inputs,
                preds,
                ids
            )
    
    def validation_epoch_end(self, outputs: List) -> None:
        map = self.metrics.compute()
        self.log_dict(map, prog_bar=True)
        print(f"mAP: {map['map']}, mAP50: {map['map_50']}")
        self.metrics.reset()

    def test_step(self, batch: Batch, batch_idx: int) -> None:
        inputs, targets, ids = batch
        preds: List[Dict[str, Tensor]] = self.net(inputs)
        keep_preds = self.__nms(preds)
        self.metrics.update(keep_preds, targets)
        return

    def test_epoch_end(self, outputs: Dict[str, Tensor]) -> None:
        map = self.metrics.compute()
        self.log_dict(map, prog_bar=False)
        print(f"mAP: {map['map']}, mAP50: {map['map_50']}")
        self.metrics.reset()
        return
    
    def __nms(self, preds: List[Dict[str, Tensor]]) -> List[Dict[str, Tensor]]:
        nms_results: List[Dict[str, Tensor]] = []
        threshold = self.config.iou_threshold
        for p in preds:
            indices = batched_nms(p['boxes'], p['scores'], p['labels'], threshold)
            keep: Dict[str, Tensor] = {}
            keep['boxes'] = p['boxes'][indices]
            keep['scores'] = p['scores'][indices]
            keep['labels'] = p['labels'][indices]
            nms_results.append(keep)

        return nms_results

    def predict_step(self, input: Tensor, batch_idx: int) -> Tensor:
        preds = self.net(input)
        return preds

    def create_model(self, num_classes: int, arch: str='resnet50') -> nn.Module:
        print(f'create model: {arch}, num classes: {num_classes}')
        ## documentation
        ## https://pytorch.org/vision/master/models/generated/torchvision.models.detection.fasterrcnn_resnet50_fpn.html#torchvision.models.detection.fasterrcnn_resnet50_fpn
        if arch == 'resnet50':
            net = detection.fasterrcnn_resnet50_fpn_v2(
                weights=detection.FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT,
                trainable_backbone_layers=3
            )
        else:  ## mobilenet v3
            net = detection.fasterrcnn_mobilenet_v3_large_fpn(
                weights=detection.FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT,
                num_claases=num_classes,
                trainable_backbone_layers=3
            )
        
        in_features = net.roi_heads.box_predictor.cls_score.in_features
        net.roi_heads.box_predictor = detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
        return net