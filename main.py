from Models.ResNet import resnet50
from Models.ViT import vit_s16
from Dataset import create_dataloader
from running import setup4training, train_one_epoch, val_one_epoch
from utils import get_parameter_num
from configs import config, update_cfg, preprocess_cfg
from log import setup_default_logging

import os
import argparse
import yaml
import logging
import random
import numpy as np
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
logger = logging.getLogger('train')

def parse_argument():

    parser = argparse.ArgumentParser(description='FGVC params')
    parser.add_argument('--config', type=str, default='')
    parser.add_argument('--local_rank', type=int, default=0)

    return parser.parse_known_args()


def create_model(config):

    name = config.model.backbone.name
    num_classes = config.model.head.num_classes
    pretrain = config.model.pretrained.pretrain
    pretrained_path = config.model.pretrained.path

    if name == 'resnet50':
        if pretrain:
            model = resnet50(pretrained=True)
            model.fc = nn.Linear(2048, num_classes)
        else:
            model = resnet50(pretrained=False)
            model.fc = nn.Linear(2048, num_classes)

    elif name == 'vit_s16':
        if pretrain:
            model = vit_s16(config, pretrained=True)
            model.head = nn.Linear(384, num_classes)
        else:
            model = vit_s16(config, pretrained=False)
            model.head = nn.Linear(384, num_classes)

    else:
        raise NotImplementedError

    return model
def main():
    setup_default_logging()
    args, config_overrided = parse_argument()
    print(args.config)
    update_cfg(config, args.config, config_overrided)
    preprocess_cfg(config, args.local_rank)

    if 'WORLD_SIZE' in os.environ:
        config.distributed = int(os.environ['WORLD_SIZE']) > 1
    else:
        config.distributed = False
    
    if config.distributed:
        config.device = 'cuda:%d' % config.local_rank
        torch.cuda.set_device(config.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        config.world_size = torch.distributed.get_world_size()
        config.rank = torch.distributed.get_rank()
        logger.info('Training in distributed mode with \
                     multiple processes, 1 GPU per process.\
                     Process {:d}, total {:d}.'.format(config.rank, config.world_size))
    else:
        config.device = 'cuda:0'
        config.world_size = 1
        config.rank = 0
        logger.info('Training with a single process on 1 GPU.')

    torch.manual_seed(config.seed + config.rank)
    np.random.seed(config.seed + config.rank)
    random.seed(config.seed + config.rank)
    

    model = create_model(config)
    model = model.cuda()

    train_loader, val_loader = create_dataloader(config, logger)
    
    config.data.batches = len(train_loader)

    optimizer, scheduler, scaler = setup4training(model, config)

    num_epochs = config.optim.epochs + config.optim.warmup_epochs

    if config.distributed:
        model = DistributedDataParallel(model, device_ids=[config.local_rank])

    output_dir = None
    eval_metric = config.eval_metric
    best_metric = None
    best_epoch = None
    checkpoint_saver = None

    if config.local_rank == 0:
        output_dir = config.output_dir
        with open(os.path.join(config.output_dir, 'config.yaml'), 'w') as f:
            yaml.dump(config.to_dict(), f)
    
    if config.loss.name == 'ce':
        criterion = nn.CrossEntropyLoss(label_smoothing=config.loss.label_smoothing)
    else:
        raise NotImplementedError

    for epoch in range(num_epochs):

        if config.distributed and hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)
        
        train_metrics = train_one_epoch(epoch, model, train_loader,
                                        optimizer, criterion, scheduler,
                                        scaler, config, logger)
        eval_metrics = val_one_epoch(model, val_loader, criterion, config, logger)



if __name__ == "__main__":
    main() 



