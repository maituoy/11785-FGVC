from numpy import identity
import torch
import torch.nn as  nn
import torch.nn.functional as F
from timm.models.layers import DropPath
from torch.hub import load_state_dict_from_url

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=None, drop_path=0, stride=1):
        super().__init__()

        self.expansion = 1
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

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

        if self.downsample is not None:
            identity = self.downsample(identity)
        
        x = self.drop_path(x) + identity
        x = self.relu(x)

        return x

class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=None, drop_path=0, stride=1):
        super().__init__()

        self.expansion = 4
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, self.expansion*out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*out_channels)

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
    def __init__(self, block, num_layers=18, dims=[64, 128, 256, 512], num_classes=7000, in_channels=3, drop_path=0.0):
        super().__init__()

        assert num_layers in [18,50], f'Only support 18 and 50 layers'
        if num_layers == 18:
            depths = [2,2,2,2]
            self.expansion = 1
        elif num_layers == 50:
            depths = [3,4,6,3]
            self.expansion = 4

        self.inplanes = 64

        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        drop_path_rates = [x.item() for x in torch.linspace(0, drop_path, sum(depths))] 

        self.layer1 = self.make_layers(block, depths[0], dims[0], drop_path=drop_path_rates[0:depths[0]])
        self.layer2 = self.make_layers(block, depths[1], dims[1], drop_path=drop_path_rates[depths[0]:sum(depths[:2])], stride=2)
        self.layer3 = self.make_layers(block, depths[2], dims[2], drop_path=drop_path_rates[sum(depths[:2]):sum(depths[:3])], stride=2)
        self.layer4 = self.make_layers(block, depths[3], dims[3], drop_path=drop_path_rates[sum(depths[:3]):sum(depths[:4])], stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))

        if num_layers == 18:
            self.fc = nn.Linear(512 * self.expansion, num_classes)
        
        elif num_layers == 50:
            # self.fc = nn.Sequential(nn.Linear(512 * self.expansion, 1024),
            #                         nn.Dropout(0.25),
            #                         nn.Linear(1024, num_classes))
            self.fc = nn.Linear(2048, num_classes)
                

    
    def forward(self, x, return_feats=False):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.max_pool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        feats = x
        x = self.fc(feats)

        if return_feats:
            return feats
        else:
            return x

    def make_layers(self, block, depth, dim, drop_path=[], stride=1):

        layers = []

        if stride != 1 or self.inplanes != dim*self.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, dim*self.expansion, kernel_size=1, stride=stride, bias=False),
                                   nn.BatchNorm2d(dim*self.expansion))
        else:
            downsample = None
        
        
        layers.append(block(self.inplanes, dim, downsample, drop_path=drop_path[0], stride=stride))
        self.inplanes = dim*self.expansion

        for i in range(depth-1):
            layers.append(block(self.inplanes, dim, drop_path=drop_path[i+1]))

        return nn.Sequential(*layers)

def resnet50(pretrained=False, progress=True, **kwargs):

    model = ResNet(Bottleneck, 50, num_classes=1000, **kwargs)

    if pretrained:
        state_dict = load_state_dict_from_url("https://download.pytorch.org/models/resnet50-0676ba61.pth", progress=progress)
        model.load_state_dict(state_dict)

    return model














        