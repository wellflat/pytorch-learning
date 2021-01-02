
from typing import Dict
import torch
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from utils import progress_bar


class FasterRCNNTrainer:
    net: FasterRCNN
    optimizer: optim.SGD
    scheduler: optim.lr_scheduler.CosineAnnealingLR
    start_epoch: int
    tensorboard_writer: SummaryWriter


    def __init__(self, conf: dict) -> None:
        super().__init__()
        self.net = self.__prepare_net(conf.get('classes', 3))
        param = [p for p in self.net.parameters() if p.requires_grad]
        #print(len(param))
        self.optimizer = optim.SGD(
            param,
            lr=conf.get('base_lr', 0.01),
            momentum=conf.get('momentum', 0.9),
            weight_decay=5e-4
        )
        #self.schedular = StepLR(self.optimizer, step_size=25, gamma=0.1)
        self.schedular = CosineAnnealingLR(self.optimizer, T_max=200)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        torch.backends.cudnn.benchmark = True
        self.net.to(self.device)
        self.best_acc = 0.0
        self.start_epoch = 0
        self.tensorboard_writer = SummaryWriter('./tensorboard_logs')


    def train(self, loaders: Dict[str, DataLoader], epochs: int, resume: bool=False) -> None:
        self.start_epoch = 0
        if resume:
            self.load_checkpoint()
        
        self.net.train()
        device = self.device
        
        for epoch in range(self.start_epoch, self.start_epoch + epochs):
            print('\nEpoch: %d' % (epoch,))
            for phase in ['train', 'val']:
                if phase == 'train':
                    self.net.train()
                else:
                    self.net.eval()

                running_loss = 0.0
                for batch_idx, (inputs, targets, image_ids) in enumerate(loaders[phase]):
                    inputs = list(input.to(device) for input in inputs)
                    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                    if phase == 'train':
                        loss_dict = self.net(inputs, targets)
                        # {'loss_classifier': tensor(0.9050, grad_fn=<NllLossBackward>), 'loss_box_reg': tensor(0.1463, grad_fn=<DivBackward0>), 'loss_objectness': tensor(0.0120, grad_fn=<BinaryCrossEntropyWithLogitsBackward>), 'loss_rpn_box_reg': tensor(0.0026, grad_fn=<DivBackward0>)} 
                        loss = sum(loss for loss in loss_dict.values())
                        #print(loss, loss.item())
                        # tensor(1.0659, grad_fn=<AddBackward0>)
                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()
                        running_loss += loss.item()
                        progress_bar(batch_idx, len(loaders[phase]), 'Loss: %.5f'
                             % (loss.item(),))
                    else:
                        pass
                        #outputs = self.net(inputs)
                        #print(outputs)
                        '''
                        [{'boxes': tensor([[119.1820,  45.1547, 324.0118, 240.4593],
                                           [100.1194,  93.6662, 349.3770, 330.9216],
                                           [241.9631,  69.2862, 275.3797, 118.3835],
                                           [177.3464, 127.1679, 327.4431, 272.7445]], device='cuda:0', grad_fn=<StackBackward>),
                          'labels': tensor([1, 1, 1, 1], device='cuda:0'),
                          'scores': tensor([0.9971, 0.0896, 0.0844, 0.0558], device='cuda:0', grad_fn=<IndexBackward>)},
                         {'boxes': ... ]
                         '''
                    
                if phase == 'train':
                    epoch_loss = running_loss / len(loaders[phase])
                    self.schedular.step()
                    print('lr: %f' % (self.schedular.get_last_lr()[0]))
                    self.tensorboard_writer.add_scalar('training loss', epoch_loss, epoch)
                #else:
                #    self.tensorboard_writer.add_scalar('validation accuracy', epoch_acc, epoch)
                
                if phase == 'val':
                    print('saving checkpoint...')
                    state = {
                        'net': self.net.state_dict(),
                        'epoch': epoch
                    }
                    torch.save(state, './checkpoint/model.pth')                


    def test(self, loader: DataLoader) -> None:
        self.net.eval()
        device = self.device

        with torch.no_grad():
            for batch_idx, (inputs, targets, image_ids) in enumerate(loader):
                inputs = list(input.to(device) for input in inputs)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                outputs = self.net(inputs)
                for output in outputs:
                    keep = torchvision.ops.batched_nms(
                        output['boxes'],
                        output['scores'],
                        output['labels'],
                        0.5
                    )
                    print(output['boxes'][keep[0].item()])


    def save(self, path: str) -> None:
        torch.save(self.net.state_dict(), path)


    def load(self, path: str) -> None:
        self.net.load_state_dict(torch.load(path))


    def load_checkpoint(self) -> None:
        checkpoint = torch.load('./checkpoint/model.pth')
        self.net.load_state_dict(checkpoint['net'])
        self.start_epoch = checkpoint['epoch'] + 1


    def __prepare_net(self, num_classes: int) -> FasterRCNN:
        net = fasterrcnn_resnet50_fpn(pretrained=True)
        in_features = net.roi_heads.box_predictor.cls_score.in_features
        net.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        return net
