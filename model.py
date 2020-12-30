
from torch import nn
from torch.nn import functional as F
from torchvision import models
from torchvision.models.resnet import ResNet


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def prepare_resnet(output_dims: int, pretrained: bool=True) -> ResNet:
    resnet = models.resnet18(pretrained=pretrained)
    if pretrained:
        for p in resnet.parameters():
            p.requires_grad = False
    
    fc_input_dim = resnet.fc.in_features
    resnet.fc = nn.Linear(fc_input_dim, output_dims)
    return resnet


