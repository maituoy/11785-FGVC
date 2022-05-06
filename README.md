# 11785-FGVC
Repo for the course project of 11785 Introduction to Deep Learning. In this project, we are trying to test which component proposed in the ConvNeXt paper is essential for FGVC tasks.

## Usage
To run the calculation with multi-gpus:

`python3 -m torch.distributed.launch --nproc_per_node=8 --master_port=10051 main.py --config /PATH/TO/YOUR/CONFIGS`

To run the calculation with single-gpu:

`python3 -m torch.distributed.launch --nproc_per_node=1 --master_port=10051 main.py --config /PATH/TO/YOUR/CONFIGS`

## Datasets
### ImageNet 1K
ImageNet 1K is used for pretraining the model with an image size of 112x112. Random augmentation, mixup, cutmix, and label smoothing are enabled in the pretraining.
### CUB2011
Modified models are fine-tuned and tested on the CUB2011 dataset later with an image size of 224x224.
### Standford Dogs
Modified models are fine-tuned and tested on the Standford Dogs dataset later with an image size of 224x224.

In fine-tuning, only the random flip and random crop are used in the training dataset while center crop is used in the validation and test set.

## Accuracy for baseline models

### Pratrained Models
| Model | Image size | Input size | #params | Top 1 Acc. | Dataset | Pretrained |
|:---:|:---:|:---:|:---:| :---:|:---:|:---:|
| Resnet50 | 256x256 | 224x224 | 25M | 81.1% | CUB2011 | True |
| ConvNeXt-T | 256x256 | 224x224 | 27M | 81.2% | CUB2011 | True |
| ViT-S16 | 256x256 | 224x224 | 22M | 72.4% | CUB2011 | True |
| Resnet50 | 256x256 | 224x224 | 25M | 84.7% | Standford Dogs | True |
| ConvNeXt-T | 256x256 | 224x224 | 27M  | 89.4% | Standford Dogs | True |
| ViT-S16 | 256x256 | 224x224 | 22M | 86.5% | Standford Dogs | True |
|  |  |  |  |  |  |  |
| Resnet50 | 512x512 | 448x448 | 25M | 86.2% | CUB2011 | True |
| ConvNeXt-T | 512x512 | 448x448 |  27M |  83.3% | CUB2011 | True |
| ViT-S16 | 512x512 | 448x448 | 22M | 79.6% | CUB2011 | True |
| Resnet50 | 512x512 | 448x448 | 25M | 85.3% | Standford Dogs | True |
| ConvNeXt-T | 512x512 | 448x448 |  27M | 87.5% | Standford Dogs | True |
| ViT-S16 | 512x512 | 448x448 | 22M | 90.2% | Standford Dogs | True |

## Modified models

### Model I
ResNet50 (Baseline model)
### Model II
Stage ratio: [3,4,6,3] -> [3,3,9,3] 

Stem: [Conv2d(kernel size 7, stride 2),MaxPool2d(kernel size 3, stride 2)] -> Conv2d(kernel size 4, stride 4)

Normal conv. layer -> Depthwise Conv. layer

Block width: [63,128,256,512] -> [96, 192, 384, 768]
### Model III
Inverted bottleneck
### Model IV
Depthwise layer moved up

Kernel size: 3 -> 7
### Model V
ReLU -> GELU

BN -> LN

fewer norms and acts

## Pretraining accuracy on ImageNet 1K
| Model | Image size | Input size | Top 1 Acc.| Top 1 Acc. in Paper|
|:---:|:---:|:---:|:---:| :---:|
| Model I  | 128x128 | 112x112 | 75.0% | 78.8% |
| Model II | 128x128 | 112x112 | 76.9% | 79.5% |
| Model III| 128x128 | 112x112 | 77.0% | 80.5% |
| Model IV | 128x128 | 112x112 | 76.9% | 80.6% |
| Model V  | 128x128 | 112x112 | 77.5% | 82.0% |

Smaller accuracy is due to the smaller image size we used in our pretraining.

## Accuracy for modified models
| Model | Image size | Input size | #params | Top 1 Acc. | Dataset | Pretrained |
|:---:|:---:|:---:|:---:| :---:|:---:|:---:|
| Model I  | 256x256 | 224x224 | 23917832 | 75.2% | CUB2011 | True |
| Model II | 256x256 | 224x224 | 31302728 | 79.1% | CUB2011 | True |
| Model III| 256x256 | 224x224 | 27568232 | 79.4% | CUB2011 | True |
| Model IV | 256x256 | 224x224 | 27614600 | 76.7% | CUB2011 | True |
| Model V  | 256x256 | 224x224 | 27973928  | 79.5% | CUB2011 | True |
|  |  |  |  |  |  |  |
| Model I  | 256x256 | 224x224 | 23917832 | 82.5% | Standford Dogs | True |
| Model II | 256x256 | 224x224 | 31302728 | 85.2% | Standford Dogs | True |
| Model III| 256x256 | 224x224 | 27568232 | 85.7% | Standford Dogs | True |
| Model IV | 256x256 | 224x224 | 27614600 | 84.8% | Standford Dogs | True |
| Model V  | 256x256 | 224x224 | 27973928  | 88.4% | Standford Dogs | True |
