
from typing import Dict, List
import torch
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops.boxes import box_iou
import tqdm

from utils import progress_bar


class FasterRCNNDetector:
    net: FasterRCNN
    optimizer: optim.SGD
    scheduler: optim.lr_scheduler.CosineAnnealingLR
    start_epoch: int
    log_writer: SummaryWriter


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
        self.best_acc = 0.0
        self.start_epoch = 0
        self.log_writer = SummaryWriter('./tensorboard_logs')


    def fit(self, loaders: Dict[str, DataLoader], epochs: int, resume: bool=False) -> None:
        self.start_epoch = 0
        if resume:
            self.load_checkpoint()
        val_loss = self.__validate(loaders['val'])
        return
        for epoch in range(self.start_epoch, self.start_epoch + epochs):
            print('\nEpoch: %d' % (epoch,))
            train_loss = self.__train(loaders['train'])
            epoch_loss = train_loss / len(loaders['train'])
            print('lr: %f' % (self.schedular.get_last_lr()[0]))
            self.log_writer.add_scalar('training loss', epoch_loss, epoch)

            val_loss = self.__validate(loaders['val'])

            self.schedular.step()

            print('saving checkpoint...')
            state = {
                'net': self.net.state_dict(),
                'epoch': epoch
            }
            torch.save(state, './checkpoint/model.pth') 
              

    def __train(self, loader: DataLoader) -> float:
        self.net.train()
        running_loss = 0.0
        for batch_idx, (inputs, targets, image_ids) in enumerate(loader):
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
            progress_bar(batch_idx, len(loader), 'Loss: %.5f' % (loss.item(),))
        
        return running_loss
        

    def __validate(self, loader: DataLoader):
        self.net.eval()
        running_loss = 0.0
        total = 0
        correct = 0
        i = 0
        for batch_idx, (inputs, targets, image_ids) in enumerate(loader):
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
            
            for output, target in zip(outputs, targets):
                if len(output['boxes']) > 0:
                    print(len(output['boxes']), len(target['boxes']))
                    print(output['scores'])
                    iou = box_iou(output['boxes'], target['boxes'])
                    print(iou)
            
            if i > 5:
                import sys
                sys.exit(0)
            i += 1
                    



    def test(self, loader: DataLoader, category) -> None:
        self.net.eval()
        device = self.device

        categories = ['background', 'dog', 'cat']

        num_nms = 0
        num_no_detect = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets, image_ids) in enumerate(loader):
                inputs = list(input.to(device) for input in inputs)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                outputs = self.net(inputs)
                for input, output, image_id in zip(inputs, outputs, image_ids):
                    keep = torchvision.ops.batched_nms(
                        output['boxes'],
                        output['scores'],
                        output['labels'],
                        0.3
                    )
                    #print(output['boxes'], keep)
                    num_detected = len(output['boxes'])
                    labels = [category[label] for label in output['labels'].tolist()]
                    if not num_detected == len(keep):
                        num_nms += 1
                    if num_detected > 0:
                        bbox = output['boxes'][keep[0].item()]
                        score = output['scores'][keep[0].item()]
                        label = output['labels'][keep[0].item()]
                        print(f'label: {categories[label.item()]}, score: {score.item()}')
                        tag = f'{image_id}'
                        self.log_writer.add_image_with_boxes(
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


    def save(self, path: str) -> None:
        torch.save(self.net.state_dict(), path)


    def load(self, path: str) -> None:
        self.net.load_state_dict(torch.load(path, map_location=self.device))


    def load_checkpoint(self, path: str='./checkpoint/model.pth') -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.net.load_state_dict(checkpoint['net'])
        self.start_epoch = checkpoint['epoch'] + 1


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
