
from typing import Dict
import torch
from torch import nn, optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torch.utils.tensorboard import SummaryWriter
from torchvision.models.resnet import ResNet
from torch.utils.data import DataLoader
import model
from utils import progress_bar
from clearml import Logger

class Trainer:
    net: ResNet
    criterion: nn.CrossEntropyLoss
    optimizer: optim.SGD
    scheduler: optim.lr_scheduler.StepLR
    best_acc: float
    start_epoch: int
    #tensorboard_writer: SummaryWriter
    logger: Logger

    def __init__(self, conf: Dict, logger: Logger, is_finetune: bool=True) -> None:
        super().__init__()
        torch.backends.cudnn.benchmark = True
        self.net = model.prepare_resnet(conf.get('classes', 10), pretrained=is_finetune)
        self.criterion = nn.CrossEntropyLoss()
        if is_finetune:
            param = self.net.fc.parameters()
        else:
            param = self.net.parameters()
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
        #self.tensorboard_writer = SummaryWriter('./tensorboard_logs')
        self.logger = logger
        
    
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

                for batch_idx, (inputs, targets) in enumerate(loaders[phase]):
                    inputs, targets = inputs.to(device), targets.to(device)
                    self.optimizer.zero_grad()
                    outputs = self.net(inputs)
                    loss = self.criterion(outputs, targets)
                    if phase == 'train':
                        loss.backward()
                        self.optimizer.step()
                    
                    running_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()
                    accuracy = 100.*correct/total

                    progress_bar(batch_idx, len(loaders[phase]), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (running_loss/(batch_idx+1), accuracy, correct, total))
                
                epoch_acc = 100.*correct / len(loaders[phase].dataset)
                epoch_loss = running_loss / len(loaders[phase])

                if phase == 'train':
                    self.schedular.step()
                    print('lr: %f' % (self.schedular.get_last_lr()[0]))
                    #self.tensorboard_writer.add_scalar('training loss', epoch_loss, epoch)
                    self.logger.report_scalar('training loss', 'epochs', epoch_loss, epoch)
                else:
                    #self.tensorboard_writer.add_scalar('validation accuracy', epoch_acc, epoch)
                    self.logger.report_scalar('validation accuracy', 'epochs', epoch_acc, epoch)
                
                if phase == 'val' and epoch_acc > self.best_acc:
                    print('saving checkpoint...')
                    state = {
                        'net': self.net.state_dict(),
                        'acc': epoch_acc,
                        'epoch': epoch
                    }
                    torch.save(state, './checkpoint/model.pth')
                    self.best_acc = epoch_acc
                
    
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

