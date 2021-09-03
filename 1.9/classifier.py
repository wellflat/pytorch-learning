
from typing import Any, Dict, Tuple
import os
import torch
from torch import nn, optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from tqdm.autonotebook import tqdm
from alexnet import AlexNet


class Classifier:
    net: AlexNet
    criterion: nn.CrossEntropyLoss
    optimizer: optim.SGD
    scheduler: optim.lr_scheduler.StepLR


    def __init__(self, conf: Dict[str, Any]):
        super().__init__()
        self.net = AlexNet(conf.get('num_classes', 10))
        self.criterion = nn.CrossEntropyLoss()

        self.optimizer = optim.SGD(
            self.net.parameters(),
            lr=conf.get('base_lr', 0.01),
            momentum=conf.get('momentum', 0.9),
            weight_decay=5e-4
        )
        self.schedular = StepLR(self.optimizer, step_size=25, gamma=0.1)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        torch.backends.cudnn.benchmark = True if torch.cuda.is_available() else False
        self.net.to(self.device)
        self.best_acc = 0.0
        self.start_epoch = 0
                
    
    def fit(self, loaders: Dict[str, DataLoader], epochs: int, resume: bool=False) -> None:
        best_acc = 0.0
        start_epoch = 0
        if resume:
            start_epoch, best_acc = self.load_checkpoint()
        
        progress = tqdm(
            range(start_epoch, start_epoch + epochs),
            total=epochs, initial=start_epoch, ncols=120, position=0
        )
        progress.set_description('Epoch')
        for epoch in progress:
            loss = self.__train(loaders['train'])
            #self.logger.add_scalar('training loss', loss, epoch)
            val_acc = self.__validate(loaders['val'])
            #self.logger.add_scalar('validation accuracy', val_acc, epoch)
            self.schedular.step()
            lr = self.schedular.get_last_lr()[0]
            #self.logger.add_scalar('learning rate', lr, epoch)
            
            if val_acc > best_acc:
                tqdm.write('saving checkpoint...')
                best_acc = val_acc
                state = {
                    'net': self.net.state_dict(),
                    'acc': best_acc,
                    'epoch': epoch
                }
                self.save_checkpoint(state)
            
            progress.set_postfix({'loss': loss, 'acc': val_acc, 'lr': lr})


    def __train(self, loader: DataLoader) -> float:
        self.net.train()
        running_loss = 0.0
        correct = 0
        total = 0
        progress = tqdm(enumerate(loader), total=len(loader), leave=False, ncols=120, position=1)
        progress.set_description('Train')
        for batch_idx, (inputs, targets) in progress:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.net(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            accuracy = 100.*correct/total

            progress.set_postfix({
                'loss': (running_loss/(batch_idx+1)),
                'acc': accuracy
            })

        epoch_loss = running_loss / len(loader)
        return epoch_loss
    

    def __validate(self, loader: DataLoader) -> float:
        self.net.eval()
        correct = 0
        total = 0
        progress = tqdm(enumerate(loader), total=len(loader), leave=False, ncols=120, position=2)
        progress.set_description('Val  ')
        for batch_idx, (inputs, targets) in progress:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            with torch.no_grad():
                outputs: torch.Tensor = self.net(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            accuracy = 100.*correct/total

            progress.set_postfix({'acc': accuracy})
                
        epoch_accuracy = 100.*correct / len(loader.dataset)
        return epoch_accuracy


    def test(self, loader: DataLoader) -> None:
        self.net.eval()
        device = self.device
        correct = 0
        total = 0

        progress = tqdm(enumerate(loader), total=len(loader), ncols=120)
        progress.set_description('Test')
        for batch_idx, (inputs, targets) in progress:
            inputs, targets = inputs.to(device), targets.to(device)
            with torch.no_grad():
                outputs: torch.Tensor = self.net(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            accuracy = 100.*correct/total

            progress.set_postfix({'acc': accuracy})
        
        test_accuracy = 100.*correct / len(loader.dataset)
        print(f'test accuracy: {test_accuracy}')


    def predict(self, x: torch.Tensor) -> torch.Tensor:
        self.net.eval()
        x.to(self.device)
        return self.net(x)


    def load(self, path: str) -> None:
        self.net.load_state_dict(torch.load(path, map_location=self.device))
    

    def save(self, path: str) -> None:
        torch.save(self.net.state_dict(), path)


    def load_checkpoint(self) -> Tuple[int, float]:
        filepath = f'./checkpoint/{self.__class__.__name__}_model.pth'
        checkpoint = torch.load(filepath, map_location=self.device)
        self.net.load_state_dict(checkpoint['net'])
        return (int(checkpoint['epoch']), checkpoint['acc'])


    def save_checkpoint(self, state: Dict[str, Any]) -> None:
        os.makedirs('./checkpoint', exist_ok=True)
        filepath = f'./checkpoint/{self.__class__.__name__}_model.pth'
        torch.save(state, filepath)



