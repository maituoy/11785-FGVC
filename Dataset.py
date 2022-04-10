import torchvision
from torch.utils.data import Dataset, DataLoader, Subset, RandomSampler, DistributedSampler, SequentialSampler
from torchvision.datasets.utils import download_file_from_google_drive
from torchvision import transforms

from PIL import Image
import pandas as pd
import os
import tarfile
from scipy.io import loadmat

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
    
    else:
        raise NotImplementedError
    
    if config.data.sampler.name is None:
        train_sampler = RandomSampler(train_dataset) if config.world_size == 1 else DistributedSampler(train_dataset)
        train_loader = DataLoader(train_dataset,
                                  sampler=train_sampler,
                                  batch_size=config.data.batch_size,
                                  num_workers=config.data.num_workers,
                                  drop_last=True,
                                  pin_memory=False)

    else:
        raise NotImplementedError
    
    test_sampler = SequentialSampler(test_dataset) if config.world_size == 1 else DistributedSampler(test_dataset)
    test_loader = DataLoader(test_dataset,
                             sampler=test_sampler,
                             batch_size=config.data.batch_size,
                             num_workers=config.data.num_workers,
                             pin_memory=False)
    
    return train_loader, test_loader