import sys
sys.path.append('../')

from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model
import tarfile
from utils import LayerNorm

class Bottleneck(nn.Module):
    def __init__(self, dim, drop_path=0, stride=1):
        super().__init__()

        self.expansion = 4
        
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=7, stride=1, padding=3, bias=False, groups=dim)
        self.conv2 = nn.Linear(dim, dim * self.expansion)
        self.conv3 = nn.Linear(dim * self.expansion, dim)

        self.layer_norm = LayerNorm(dim, eps=1e-6, data_format="channels_first")

        self.gelu = nn.GELU()
        
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        
    def forward(self, x):

        identity = x.clone()

        x = self.conv1(x)
        x = self.layer_norm(x)

        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.conv2(x) # linear

        x = self.gelu(x)

        x = self.conv3(x)  # linear
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = self.drop_path(x) + identity

        
        return x 

class ResNet(nn.Module):
    def __init__(self, block=Bottleneck, 
                 in_chans=3, 
                 depths = [3,4,6,3], 
                 dims=[64, 128, 256, 512], 
                 num_classes=1000, 
                 in_channels=3, 
                 drop_path_rate=0., 
                 layer_scale_init_value=1e-6, 
                 head_init_scale=1.,
                 ):
        super().__init__()

        self.expansion = 4
        self.inplanes = 64
        
        # 4 downsample layers
        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
            # nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            # LayerNorm(64, eps=1e-6, data_format="channels_first"),#nn.BatchNorm2d(64),
            # nn.GELU(), #nn.ReLU(),
            # nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.downsample_layers.append(stem)
        # the rest 3 downsampling layers
        for i in range(3):
            downsample_layer = nn.Sequential(
                    LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)
        
        
        # 4 stages
        dp_rates=[x.item() for x in torch.linspace(drop_path_rate, drop_path_rate, sum(depths))] 
        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[block(dim=dims[i], drop_path=dp_rates[cur + j]) for j in range(depths[i])])
            self.stages.append(stage)
            cur += depths[i]
      
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.norm = nn.LayerNorm(dims[-1], eps=1e-6) # final norm layer
        
        self.fc = nn.Linear(dims[-1], num_classes)

        # weight as used in convnext code
        #self.apply(self._init_weights)
        self.fc.weight.data.mul_(head_init_scale)
        self.fc.bias.data.mul_(head_init_scale)
    
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)
    
    def forward(self, x):

        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        
        #x = self.avgpool(x)
        x = self.norm(x.mean([-2, -1])) 
        
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        
        return x


def resnet50_c12345(num_classes=1000, **kwargs): # c12

    model = ResNet(Bottleneck, num_classes=num_classes, **kwargs)

    return model