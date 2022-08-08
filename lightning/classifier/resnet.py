from torch import nn
from torchvision.models.resnet import ResNet, BasicBlock


class MNISTResNet(ResNet):
    def __init__(self):
        super().__init__(BasicBlock, [2,2,2,2], num_classes=10)
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=1, padding=3, bias=False)


def calc_size(hin:int) -> int:
    padding = 3
    kernel = 7
    stride = 1
    hout = (hin + 2*padding - kernel) / stride + 1
    return hout


if __name__ == '__main__':
    from torchsummary import summary
    model = MNISTResNet()
    print(model)
    print(summary(model, (1,28,28)))
    print(calc_size(28))
    
    
    