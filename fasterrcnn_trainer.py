
from typing import Dict
import torch
from torch import nn, optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torch.utils.tensorboard import SummaryWriter
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN
from torch.utils.data import DataLoader
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from utils import progress_bar


class FasterRCNNTrainer:
    net: FasterRCNN
    criterion: nn.CrossEntropyLoss
    optimizer: optim.SGD
    scheduler: optim.lr_scheduler.StepLR
    best_acc: float
    start_epoch: int
    tensorboard_writer: SummaryWriter

    def __init__(self, conf: dict) -> None:
        super().__init__()
        torch.backends.cudnn.benchmark = True
        self.net = self.__prepare_net(conf.get('classes', 3))
        self.criterion = nn.CrossEntropyLoss()
        param = [p for p in self.net.parameters() if p.requires_grad]
        #print(param)
        self.optimizer = optim.SGD(
            param,
            lr=conf.get('base_lr', 0.01),
            momentum=conf.get('momentum', 0.9),
            weight_decay=5e-4
        )
        #self.schedular = StepLR(self.optimizer, step_size=25, gamma=0.1)
        self.schedular = CosineAnnealingLR(self.optimizer, T_max=200)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)
        self.best_acc = 0.0
        self.start_epoch = 0
        self.tensorboard_writer = SummaryWriter('./tensorboard_logs')
    
    def train(self, loaders: Dict[str, DataLoader], epochs: int, resume: bool=False) -> None:
        self.best_acc = 0.0
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
                correct = 0
                total = 0

                for batch_idx, (inputs, targets, image_ids) in enumerate(loaders[phase]):
                    inputs = list(input.to(device) for input in inputs)
                    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                    loss_dict = self.net(inputs, targets)
                    loss = sum(loss for loss in loss_dict.values())
                    if phase == 'train':
                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()
                    
                    running_loss += loss.item()
                    #_, predicted = outputs.max(1)
                    #total += targets.size(0)
                    #correct += predicted.eq(targets).sum().item()
                    #accuracy = 100.*correct/total

                    progress_bar(batch_idx, len(loaders[phase]), 'Loss: %.3f'
                         % (running_loss/(batch_idx+1),))
                    progress_bar(batch_idx, len(loaders[phase]), 'Loss: %.3f'
                         % (loss.item(),))
                
                #epoch_acc = 100.*correct / len(loaders[phase].dataset)
                epoch_loss = running_loss / len(loaders[phase])

                if phase == 'train':
                    self.schedular.step()
                    print('lr: %f' % (self.schedular.get_last_lr()[0]))
                    self.tensorboard_writer.add_scalar('training loss', epoch_loss, epoch)
                #else:
                #    self.tensorboard_writer.add_scalar('validation accuracy', epoch_acc, epoch)
                '''
                if phase == 'val' and epoch_acc > self.best_acc:
                    print('saving checkpoint...')
                    state = {
                        'net': self.net.state_dict(),
                        'acc': epoch_acc,
                        'epoch': epoch
                    }
                    torch.save(state, './checkpoint/model.pth')
                    self.best_acc = epoch_acc
                '''

    def test(self, loader: DataLoader) -> None:
        self.net.eval()
        device = self.device
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(loader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = self.net(inputs)
                loss = self.criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                progress_bar(batch_idx, len(loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
                
                
    
    def save(self, path: str) -> None:
        torch.save(self.net.state_dict(), path)
    
    def load(self, path: str) -> None:
        self.net.load_state_dict(torch.load(path))
    
    def load_checkpoint(self) -> None:
        checkpoint = torch.load('./checkpoint/model.pth')
        self.net.load_state_dict(checkpoint['net'])
        self.best_acc = checkpoint['acc']
        self.start_epoch = checkpoint['epoch']
        print(self.start_epoch, self.best_acc)

    def __prepare_net(self, num_classes: int) -> FasterRCNN:
        net = fasterrcnn_resnet50_fpn(pretrained=True)
        in_features = net.roi_heads.box_predictor.cls_score.in_features
        net.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        return net

