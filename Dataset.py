import torchvision
from torch.utils.data import Dataset, DataLoader, Subset, RandomSampler, DistributedSampler, SequentialSampler
from torchvision.datasets.utils import download_file_from_google_drive
from torchvision import transforms, datasets

from PIL import Image
import pandas as pd
import os
import tarfile
from scipy.io import loadmat

from timm.data.constants import \
    IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from timm.data import create_transform

class CUB2011(Dataset):
    def __init__(self, root, transform, train=True, extract=False):

        self.root = root

        if extract:
            self.extractfile()

        img_folder = os.path.join(root, 'images')
        split_file = os.path.join(root, 'train_test_split.txt')

        fullset = torchvision.datasets.ImageFolder(root=img_folder, transform=transform)

        split_table = pd.read_csv(split_file, delimiter=' ', header=None, names=['id','split'])

        if train:
          idxes = split_table.index[split_table['split'] == 1].tolist()
        else:
          idxes = split_table.index[split_table['split'] == 0].tolist()
        
        self.dataset = Subset(fullset, idxes)
        self.transforms = fullset.transforms
    

    def extractfile(self):

        if not os.path.exists(os.path.join(self.root, 'CUB_200_2011.tar')):
            raise RuntimeError('File not found!')

        else:
            with tarfile.open(os.path.join(self.root, 'CUB_200_2011.tar'), 'r') as tar:
                tar.extractall(path=self.root)
  
    def __len__(self):
        return len(self.dataset)
  
    def __getitem__(self, idx):
        return self.dataset[idx][0], self.dataset[idx][1]
      

class StandfordDog(Dataset):
    def __init__(self, root, transform, train=True, extract=False):

        self.root = root

        if extract:
            if not os.path.exists(root):
                os.system('mkdir StandfordDog')

            self.extractfile()

        lists_folder = self.root
        img_folder = os.path.join(self.root, 'Images')
        
        file_list = loadmat(os.path.join(lists_folder, 'file_list.mat'))
        train_list = loadmat(os.path.join(lists_folder, 'train_list.mat'))
        test_list = loadmat(os.path.join(lists_folder, 'test_list.mat'))

        train_idx, test_idx = self.get_split_idx(file_list, train_list, test_list)

        fullset = torchvision.datasets.ImageFolder(root=img_folder, transform=transform)

        if train:
            self.dataset = Subset(fullset, train_idx)
        else:
            self.dataset = Subset(fullset, test_idx)
        
        self.transforms = fullset.transforms

    def get_split_idx(self, file_list, train_list, test_list):

        files = [item[0][0] for item in file_list['file_list']]
        train_data = [item[0][0] for item in train_list['file_list']]
        test_data = [item[0][0] for item in test_list['file_list']]

        train_idx = []
        test_idx = []

        for i, file in enumerate(files):
            if file in train_data:
                train_idx.append(i)
            elif file in test_data:
                test_idx.append(i)

        return (train_idx, test_idx)

    def extractfile(self):

        if not os.path.exists(os.path.join(self.root, 'images.tar')):
            raise RuntimeError('File not found!')
        
        elif not os.path.exists(os.path.join(self.root, 'lists.tar')):
            raise RuntimeError('File not found!')

        else:
            with tarfile.open(os.path.join(self.root, 'images.tar'), 'r') as tar:
                tar.extractall(path=os.path.join(self.root, self.filename))

            with tarfile.open(os.path.join(self.root, 'lists.tar'), 'r') as tar:
                tar.extractall(path=os.path.join(self.root, self.filename))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx][0], self.dataset[idx][1]


def create_dataloader(config, logger):

    input_size = config.data.input_size
    img_size = config.data.image_size
    train_transforms = transforms.Compose([ transforms.Resize((img_size, img_size), Image.BILINEAR),
                                            transforms.RandomCrop((input_size, input_size)),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                                          ])
                            
    test_transforms = transforms.Compose([ transforms.Resize((img_size, img_size), Image.BILINEAR),
                                           transforms.CenterCrop((input_size, input_size)),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                                         ])
        

    if config.data.name == 'CUB2011':
        
        train_dataset = CUB2011(root=config.data.root, transform=train_transforms, train=True, extract=False)
        test_dataset = CUB2011(root=config.data.root, transform=test_transforms, train=False, extract=False)

    elif config.data.name == 'dog':

        train_dataset = StandfordDog(root=config.data.root, transform=train_transforms, train=True, extract=False)
        test_dataset = StandfordDog(root=config.data.root, transform=test_transforms, train=False, extract=False)
    
    elif config.data.name == 'imagenet1k':

        train_transforms = build_transform(True, config)
        val_transforms = build_transform(False, config)

        root_train = os.path.join(config.data.root, 'train')
        root_val = os.path.join(config.data.root, 'val')

        train_dataset = datasets.ImageFolder(root_train, transform=train_transforms)
        test_dataset = datasets.ImageFolder(root_val, transform=val_transforms)

    else:
        raise NotImplementedError
    
    if config.data.sampler.name is None:
        train_sampler = RandomSampler(train_dataset) if config.world_size == 1 else DistributedSampler(train_dataset)
        train_loader = DataLoader(train_dataset,
                                  sampler=train_sampler,
                                  batch_size=config.data.batch_size,
                                  num_workers=config.data.num_workers,
                                  drop_last=True,
                                  pin_memory=True)

    else:
        raise NotImplementedError
    
    test_sampler = SequentialSampler(test_dataset) if config.world_size == 1 else DistributedSampler(test_dataset)
    test_loader = DataLoader(test_dataset,
                             sampler=test_sampler,
                             batch_size=config.data.batch_size,
                             num_workers=config.data.num_workers,
                             pin_memory=True)
    
    return train_loader, test_loader

def build_transform(is_train, config):
    resize_im = config.data.input_size > 32
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD

    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=config.data.input_size,
            is_training=True,
            color_jitter=0.4,
            auto_augment='rand-m9-mstd0.5-inc1',
            interpolation='bicubic',
            re_prob=0.25,
            re_mode='pixel',
            re_count=1,
            mean=mean,
            std=std,
        )
        if not resize_im:
            transform.transforms[0] = transforms.RandomCrop(
                config.data.input_size, padding=4)
        return transform

    t = []
    if resize_im:
        # warping (no cropping) when evaluated at 384 or larger
        if config.data.input_size >= 384:  
            t.append(
            transforms.Resize((config.data.input_size, config.data.input_size), 
                            interpolation=transforms.InterpolationMode.BICUBIC), 
        )
            print(f"Warping {config.data.input_size} size input images...")
        else:
            size = config.data.image_size
            t.append(
                # to maintain same ratio w.r.t. 224 images
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),  
            )
            t.append(transforms.CenterCrop(config.data.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)
