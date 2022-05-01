from numpy import identity
import torch
import torch.nn as  nn
import torch.nn.functional as F
from timm.models.layers import DropPath
from torch.hub import load_state_dict_from_url

class Bottleneck(nn.Module):
    def __init__(self, dim, downsample=None, drop_path=0, stride=1):
        super().__init__()

        self.expansion = 4

        self.conv1 = nn.Conv2d(dim,  self.expansion*dim, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(self.expansion*dim)
        self.conv2 = nn.Conv2d(self.expansion*dim, self.expansion*dim, kernel_size=3, stride=stride, padding=1, bias=False, groups=self.expansion*dim)
        self.bn2 = nn.BatchNorm2d(self.expansion*dim)
        self.conv3 = nn.Conv2d(self.expansion*dim, dim, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(dim)

        self.relu = nn.ReLU()
        self.downsample = downsample
        
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
    
    def forward(self, x):

        identity = x.clone()

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)

        if self.downsample is not None:
            identity = self.downsample(identity)
        x = self.drop_path(x) + identity
        x = self.relu(x)
        
        return x 


class ResNet(nn.Module):
    def __init__(self, block, dims=[96, 192, 384, 768], num_classes=7000, in_channels=3, drop_path=0.0):
        super().__init__()


        self.expansion = 4
        self.depths = [3, 3, 9, 3]
        self.dims = dims

        self.conv1 = nn.Conv2d(in_channels, dims[0], kernel_size=4, stride=4)
        self.bn1 = nn.BatchNorm2d(dims[0])

        drop_path_rates = [x.item() for x in torch.linspace(0, drop_path, sum(self.depths))] 

        self.layer1 = self.make_layers(block, 0, drop_path=drop_path_rates[0:self.depths[0]])
        self.ds1 = nn.Conv2d(self.dims[0], self.dims[1], kernel_size=1, stride=1, bias=False)
        self.layer2 = self.make_layers(block, 1, drop_path=drop_path_rates[self.depths[0]:sum(self.depths[:2])], stride=2)
        self.ds2 = nn.Conv2d(self.dims[1], self.dims[2], kernel_size=1, stride=1, bias=False)
        self.layer3 = self.make_layers(block, 2, drop_path=drop_path_rates[sum(self.depths[:2]):sum(self.depths[:3])], stride=2)
        self.ds3 = nn.Conv2d(self.dims[2], self.dims[3], kernel_size=1, stride=1, bias=False)
        self.layer4 = self.make_layers(block, 3, drop_path=drop_path_rates[sum(self.depths[:3]):sum(self.depths[:4])], stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))

        self.fc = nn.Linear(dims[-1], num_classes)
                

    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)

        x = self.layer1(x)
        x = self.ds1(x)
        x = self.layer2(x)
        x = self.ds2(x)
        x = self.layer3(x)
        x = self.ds3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)

        x = self.fc(x)

        return x

    def make_layers(self, block, layer, drop_path=[], stride=1):

        layers = []


        downsample = nn.Sequential(nn.Conv2d(self.dims[layer], self.dims[layer], kernel_size=1, stride=stride, bias=False),
                                    nn.BatchNorm2d(self.dims[layer]))

        layers.append(block(self.dims[layer], downsample=downsample, stride=stride, drop_path=drop_path[0]))

        for i in range(self.depths[layer]-1):
            layers.append(block(self.dims[layer], downsample=None, drop_path=drop_path[i+1]))
        

        return nn.Sequential(*layers)

def resnet50_c12_new(num_classes=1000, **kwargs):

    model = ResNet(Bottleneck, num_classes=num_classes, **kwargs)

    return model
