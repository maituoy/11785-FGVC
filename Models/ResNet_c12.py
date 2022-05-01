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
# from Models.modules import LayerNorm
class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, drop_path=0, stride=1, first=False):
        super().__init__()

        self.expansion = 4
        
        #self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv1 = nn.Linear(in_channels, out_channels)

        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False, groups= out_channels)
        self.layer_norm = LayerNorm(out_channels, eps=1e-6)
        #self.conv3 = nn.Conv2d(out_channels, self.expansion*out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv3 = nn.Linear(out_channels, self.expansion*out_channels)

        self.gelu = nn.GELU()
        
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        
        self.first = first
        if first:
            self.conv_skip = nn.Conv2d(in_channels,  self.expansion*out_channels, kernel_size=1, stride=1, padding=0, bias=False)#, groups=in_channels)
        else:
            self.conv_skip = None
            
    def forward(self, x):

        identity = x.clone()
        if self.first:
            identity = self.conv_skip(x)

        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.conv1(x) # linear
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = self.conv2(x)
<<<<<<< HEAD
        x = x.permute(0, 2, 3, 1)
=======
>>>>>>> 5fb310d9c61db6da6dd70f6f8fd3087789098709
        x = self.layer_norm(x)

        x = self.gelu(x)
        x = self.conv3(x)  # linear
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = self.drop_path(x) + identity

        return x 

class ResNet(nn.Module):
    def __init__(self, block=Bottleneck, 
                       depths = [3,4,6,3], 
                       dims=[64, 128, 256, 512], 
                       num_classes=1000, 
                       in_channels=3, 
                       drop_path=0.0):
        super().__init__()

        self.expansion = 4
        self.inplanes = 64
        
        # 4 downsample layers
        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
#             nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
#             LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            LayerNorm(64, eps=1e-6, data_format="channels_first"),#nn.BatchNorm2d(64),
            nn.GELU(), #nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.downsample_layers.append(stem)
        # the rest 3 downsampling layers
        for i in range(3):
            downsample_layer = nn.Sequential(
                    LayerNorm(dims[i]*self.expansion, eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(dims[i]*self.expansion, dims[i]*self.expansion, kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)
        
        
        # 4 stages
        drop_path_rates = [x.item() for x in torch.linspace(0, drop_path, sum(depths))] 
        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        self.stage1 = self.make_stage(block, depths[0], dims[0], drop_path=drop_path_rates[0:depths[0]], stride=1)
        self.stage2 = self.make_stage(block, depths[1], dims[1], drop_path=drop_path_rates[depths[0]:sum(depths[:2])], stride=1)
        self.stage3 = self.make_stage(block, depths[2], dims[2], drop_path=drop_path_rates[sum(depths[:2]):sum(depths[:3])], stride=1)
        self.stage4 = self.make_stage(block, depths[3], dims[3], drop_path=drop_path_rates[sum(depths[:3]):sum(depths[:4])], stride=1)
        self.stages.append(self.stage1)
        self.stages.append(self.stage2)
        self.stages.append(self.stage3)
        self.stages.append(self.stage4)
        
    
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.norm = nn.LayerNorm(2048, eps=1e-6) # final norm layer
        
        self.fc = nn.Linear(2048, num_classes)
    
    def forward(self, x):

        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        
        #x = self.avgpool(x)
        x = self.norm(x.mean([-2, -1])) 
        
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)

        return x


    def make_stage(self, block, depth, dim, drop_path=[], stride=1):

        layers = []
    
        #first layer of each stage
        layers.append(block(self.inplanes, dim, drop_path=drop_path[0], stride=stride, first=True))
        self.inplanes = dim*self.expansion

        for i in range(depth-1):
            layers.append(block(self.inplanes, dim, drop_path=drop_path[i+1], stride=stride))

        return nn.Sequential(*layers)

def resnet50_c12(num_classes=1000, **kwargs): # c12

    model = ResNet(Bottleneck, num_classes=num_classes, **kwargs)

    # for name, param in model.named_parameters():
    #     if 'conv_skip' in name:
    #         param.requires_grad = False

    return model