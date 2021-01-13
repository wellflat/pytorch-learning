
from typing import Any, Dict, List, Tuple
import torch
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops.boxes import box_iou
from sklearn.metrics import average_precision_score
from tqdm import tqdm


Dataset = List[Dict[str, torch.Tensor]]


class FasterRCNNDetector:
    net: FasterRCNN
    optimizer: optim.SGD
    scheduler: optim.lr_scheduler.CosineAnnealingLR
    start_epoch: int
    logger: SummaryWriter


    def __init__(self, conf: dict) -> None:
        super().__init__()
        self.net = self.__prepare_net(conf.get('classes', 3))
        param = [p for p in self.net.parameters() if p.requires_grad]
        self.optimizer = optim.SGD(
            param,
            lr=conf.get('base_lr', 0.01),
            momentum=conf.get('momentum', 0.9),
            weight_decay=5e-4
        )
        #self.schedular = StepLR(self.optimizer, step_size=25, gamma=0.1)
        self.schedular = CosineAnnealingLR(self.optimizer, T_max=5, eta_min=0.001)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        torch.backends.cudnn.benchmark = True if torch.cuda.is_available() else False
        self.net.to(self.device)
        self.logger = SummaryWriter('./tensorboard_logs')


    def fit(self, loaders: Dict[str, DataLoader], epochs: int, resume: bool=False) -> None:
        best_ap = 0.0
        start_epoch = 0
        if resume:
            best_ap, start_epoch = self.load_checkpoint()
        
        progress = tqdm(
            range(start_epoch, start_epoch + epochs),
            total=epochs, initial=start_epoch, ncols=120, position=0
        )
        progress.set_description('Epoch')
        for epoch in progress:
            #loss = self.__train(loaders['train'])
            #self.logger.add_scalar('training loss', loss, epoch)
            ap = self.__validate(loaders['val'])
            self.logger.add_scalar('average precision', ap, epoch)
            self.schedular.step()
            lr = self.schedular.get_last_lr()[0]
            self.logger.add_scalar('learning rate', lr, epoch)

            if ap > best_ap:
                tqdm.write('saving checkpoint...')
                best_ap = ap
                state = {
                    'net': self.net.state_dict(),
                    'ap': ap,
                    'epoch': epoch
                }
                self.save_checkpoint(state)

            progress.set_postfix({'loss': loss, 'ap': ap, 'lr': lr})
            

    def __train(self, loader: DataLoader) -> float:
        self.net.train()
        running_loss = 0.0
        progress = tqdm(enumerate(loader), total=len(loader), leave=False, ncols=120, position=1)
        progress.set_description('Train')
        for batch_idx, (inputs, targets, image_ids) in progress:
            inputs = list(input.to(self.device) for input in inputs)
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets] 
            loss_dict = self.net(inputs, targets)
            # loss_dict: {'loss_classifier': tensor(0.9050, grad_fn=<NllLossBackward>), 'loss_box_reg': tensor(0.1463, grad_fn=<DivBackward0>), 'loss_objectness': tensor(0.0120, grad_fn=<BinaryCrossEntropyWithLogitsBackward>), 'loss_rpn_box_reg': tensor(0.0026, grad_fn=<DivBackward0>)} 
            loss: torch.Tensor = sum(loss for loss in loss_dict.values())
            # loss: tensor(1.0659, grad_fn=<AddBackward0>)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()
            #progress_bar(batch_idx, len(loader), 'Loss: %.5f' % (loss.item(),))

            progress.set_postfix({'loss': (running_loss/(batch_idx+1))})
        
        epoch_loss = running_loss / len(loader)
        return epoch_loss
        

    def __validate(self, loader: DataLoader) -> float:
        self.net.eval()
        correct = []
        scores = []
        ap = 0.0
        progress = tqdm(enumerate(loader), total=len(loader), leave=False, ncols=120, position=2)
        progress.set_description('Val  ')
        for batch_idx, (inputs, targets, image_ids) in progress:
            inputs = list(input.to(self.device) for input in inputs)
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
            with torch.no_grad():
                outputs = self.net(inputs)
                '''
                [{'boxes': tensor([[119.1820,  45.1547, 324.0118, 240.4593],
                                   [100.1194,  93.6662, 349.3770, 330.9216],
                                   [241.9631,  69.2862, 275.3797, 118.3835],
                                   [177.3464, 127.1679, 327.4431, 272.7445]], device='cuda:0', grad_fn=<StackBackward>),
                  'labels': tensor([1, 1, 1, 1], device='cuda:0'),
                  'scores': tensor([0.9971, 0.0896, 0.0844, 0.0558], device='cuda:0', grad_fn=<IndexBackward>)},
                 {'boxes': ... ]
                 '''
            
            pred_true, true_scores = self.__get_metrics(outputs, targets)
            #print(pred_true, true_scores)
            correct.append(pred_true)
            scores.append(true_scores)
            y_true = torch.cat(correct, dim=0)
            y_score = torch.cat(scores, dim=0)
            ap: float = average_precision_score(y_true, y_score)

            progress.set_postfix({'ap': ap})
        
        return ap
    

    def __get_metrics(self, outputs: Dataset, targets: Dataset, iou_threshold: float=0.75) -> Tuple[torch.Tensor, torch.Tensor]:
        ## one label/bbox per validation image (targets)
        correct = []
        scores = []
        for output, target in zip(outputs, targets):
            if len(target['labels']) > 1:
                continue
            if len(output['boxes']) == 0:
                continue

            true_label = target['labels'][0]
            true_boxes = output['boxes'][output['labels'] == true_label]
            true_scores = output['scores'][output['labels'] == true_label]
            iou = box_iou(true_boxes, target['boxes'])
            pred_true = iou > iou_threshold
            correct.append(pred_true.flatten())
            scores.append(true_scores)
            
        return (torch.cat(correct, dim=0), torch.cat(scores, dim=0))
    

    def test(self, loader: DataLoader, categories: List[str]) -> None:
        self.net.eval()
        device = self.device

        num_nms = 0
        num_no_detect = 0

        progress = tqdm(enumerate(loader), total=len(loader), ncols=120)
        progress.set_description('Test')
        for batch_idx, (inputs, targets, image_ids) in progress:
            inputs = list(input.to(device) for input in inputs)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            with torch.no_grad():
                outputs: torch.Tensor = self.net(inputs)
                
            for input, output, image_id in zip(inputs, outputs, image_ids):
                keep = torchvision.ops.batched_nms(
                    output['boxes'],
                    output['scores'],
                    output['labels'],
                    0.3
                )
                #print(output['boxes'], keep)
                num_detected = len(output['boxes'])
                labels = [categories[label] for label in output['labels'].tolist()]
                if not num_detected == len(keep):
                    num_nms += 1
                if num_detected > 0:
                    bbox = output['boxes'][keep[0].item()]
                    score = output['scores'][keep[0].item()]
                    label = output['labels'][keep[0].item()]
                    tqdm.write(f'label: {categories[label.item()]}, score: {score.item()}')
                    tag = f'{image_id}'
                    self.logger.add_image_with_boxes(
                        tag,
                        input,
                        output['boxes'],
                        global_step=batch_idx,
                        labels=labels
                    )
                else:
                    num_no_detect += 1
        
        print(f'num nms: {num_nms}, no detected: {num_no_detect}')
    

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        self.net.eval()
        x.to(self.device)
        return self.net(x)


    def load(self, path: str) -> None:
        #self.net.load_state_dict(torch.load(path, map_location=self.device))

        checkpoint = torch.load(path, map_location=self.device)
        self.net.load_state_dict(checkpoint['net'])
    

    def save(self, path: str) -> None:
        torch.save(self.net.state_dict(), path)


    def load_checkpoint(self) -> Tuple[int, float]:
        filepath = f'./checkpoint/{self.__class__.__name__}_model.pth'
        checkpoint = torch.load(filepath, map_location=self.device)
        self.net.load_state_dict(checkpoint['net'])
        return (checkpoint['epoch'], checkpoint['acc'])


    def save_checkpoint(self, state: Dict[str, Any]) -> None:
        filepath = f'./checkpoint/{self.__class__.__name__}_model.pth'
        torch.save(state, filepath)
    

    def __prepare_net(self, num_classes: int) -> FasterRCNN:
        net = fasterrcnn_resnet50_fpn(pretrained=True, trainable_backbone_layers=3)
        in_features = net.roi_heads.box_predictor.cls_score.in_features
        net.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        '''
        RoIHeads(
          (box_roi_pool): MultiScaleRoIAlign()
          (box_head): TwoMLPHead(
            (fc6): Linear(in_features=12544, out_features=1024, bias=True)
            (fc7): Linear(in_features=1024, out_features=1024, bias=True)
          )
          (box_predictor): FastRCNNPredictor(
            (cls_score): Linear(in_features=1024, out_features=3, bias=True)
            (bbox_pred): Linear(in_features=1024, out_features=12, bias=True)
          )
        )'''
        return net
