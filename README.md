# 11785-FGVC
Repo for the course project of 11785 Introduction to Deep Learning.

## Usage

## Accuracy for baseline models
### Plain Models
| Model | Image size | Input size | #params | Top 1 Acc. | Dataset | Pretrained |
|:---:|:---:|:---:|:---:| :---:|:---:|:---:|
| Resnet50 | 256x256 | 224x224 | 25M | 38.5% | CUB2011 | False |
| ConvNeXt-T | 256x256 | 224x224 | 27M |  | CUB2011 | False |
| ViT-S16 | 256x256 | 224x224 | 22M | 14.3% | CUB2011 | False |
| Resnet50 | 256x256 | 224x224 | 25M | 42.3% | Standford Dogs | False |
| ConvNeXt-T | 256x256 | 224x224 | 27M |  | Standford Dogs | False |
| ViT-S16 | 256x256 | 224x224 | 22M | 14.4% | Standford Dogs | False |
|  |  |  |  |  |  |  |
| Resnet50 | 512x512 | 448x448 | 25M |  | CUB2011 | False |
| ConvNeXt-T | 512x512 | 448x448 |  |  | CUB2011 | False |
| ViT-S16 | 512x512 | 448x448 | 22M |  | CUB2011 | False |
| Resnet50 | 512x512 | 448x448 | 25M |  | Standford Dogs | False |
| ConvNeXt-T | 512x512 | 448x448 |  |  | Standford Dogs | False |
| ViT-S16 | 512x512 | 448x448 | 22M |  | Standford Dogs | False |

### Pratrained Models
| Model | Image size | Input size | #params | Top 1 Acc. | Dataset | Pretrained |
|:---:|:---:|:---:|:---:| :---:|:---:|:---:|
| Resnet50 | 256x256 | 224x224 | 25M | 78.4% | CUB2011 | True |
| ConvNeXt-T | 256x256 | 224x224 | 27M | 81.2% | CUB2011 | True |
| ViT-S16 | 256x256 | 224x224 | 22M | 69.1% | CUB2011 | True |
| Resnet50 | 256x256 | 224x224 | 25M | 71.9% | Standford Dogs | True |
| ConvNeXt-T | 256x256 | 224x224 | 27M  | 89.4% | Standford Dogs | True |
| ViT-S16 | 256x256 | 224x224 | 22M | 77.2% | Standford Dogs | True |
|  |  |  |  |  |  |  |
| Resnet50 | 512x512 | 448x448 | 25M |  | CUB2011 | True |
| ConvNeXt-T | 512x512 | 448x448 |  |  | CUB2011 | True |
| ViT-S16 | 512x512 | 448x448 | 22M |  | CUB2011 | True |
| Resnet50 | 512x512 | 448x448 | 25M |  | Standford Dogs | True |
| ConvNeXt-T | 512x512 | 448x448 |  |  | Standford Dogs | True |
| ViT-S16 | 512x512 | 448x448 | 22M |  | Standford Dogs | True |
