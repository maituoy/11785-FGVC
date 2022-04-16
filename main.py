from Models.ResNet import resnet50
from Models.ViT import vit_s16
from Models.ResNet_c1 import resnet50_c1
from Dataset import create_dataloader
from running import setup4training, train_one_epoch, val_one_epoch
from utils import get_parameter_num, update_summary
from configs import config, update_cfg, preprocess_cfg
from log import setup_default_logging
from checkpoint import CheckpointSaver

from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import ModelEma
from timm.data.mixup import Mixup

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

    elif name == 'resnet50_c1':
        if pretrain:
            raise NotImplementedError
        else:
            model = resnet50_c1(num_classes=num_classes, drop_path=config.model.backbone.drop_path)
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

    model.cuda() 
    
    #-----------------------------------------------------------------------------------------------------------
    #Section for mixup & cutmix: 2 in 1 easy game
    mixup_fn = None
    mixup_active = config.train.mixup > 0 or config.train.cutmix > 0. or config.train.cutmix_minmax is not None
    if mixup_active:
        logger.info("Mixup is activated!")
        mixup_fn = Mixup(
            mixup_alpha=config.train.mixup, cutmix_alpha=config.train.cutmix, cutmix_minmax=config.train.cutmix_minmax,
            prob=config.train.mixup_prob, switch_prob=config.train.mixup_switch_prob, mode=config.train.mixup_mode,
            label_smoothing=config.train.smoothing, num_classes=config.model.head.num_classes)
            
    if mixup_fn is not None:
        #smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif config.train.smoothing > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=config.train.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    logger.info("criterion = %s" % str(criterion))
    
    #-----------------------------------------------------------------------------------------------------------

    #-----------------------------------------------------------------------------------------------------------
    ##Section for ema:
    model_ema = None
    if config.train.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEma(
            model,
            decay=config.train.model_ema_decay,
            device='cpu' if config.train.ema_force_cpu else '',
            resume='')
        logger.info("Using EMA with decay = %.8f" % config.train.model_ema_decay)    
    #-----------------------------------------------------------------------------------------------------------

    train_loader, val_loader = create_dataloader(config,logger)
    
    config.data.batches = len(train_loader)

    optimizer, scheduler, scaler = setup4training(model, config)

    num_epochs = config.optim.epochs + config.optim.warmup_epochs

    if config.distributed:
        model = DistributedDataParallel(model, device_ids=[config.local_rank],find_unused_parameters=False)

    output_dir = None
    eval_metric = config.eval_metric
    best_metric = None
    best_epoch = None
    checkpoint_saver = None
    decreasing = True if eval_metric == 'val_loss' else False   

    if config.local_rank == 0:
        output_dir = config.output_dir
        checkpoint_saver = CheckpointSaver(
            model=model, optimizer=optimizer, cfg=config, amp_scaler=scaler,  
            checkpoint_dir=output_dir, recovery_dir=output_dir, decreasing=decreasing)
        with open(os.path.join(config.output_dir, 'config.yaml'), 'w') as f:
            yaml.dump(config.to_dict(), f)

        logger.info(config.to_dict())
    
    # print(model)
    for epoch in range(num_epochs):

        if config.distributed and hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)
        train_metrics = train_one_epoch(epoch, model, train_loader,
                                        optimizer, criterion, scheduler,
                                        scaler, config, mixup_fn = mixup_fn,model_ema=model_ema)

        eval_metrics = val_one_epoch(model, val_loader, config)

        if output_dir is not None:
            update_summary(
                epoch, train_metrics, eval_metrics, os.path.join(output_dir, 'summary.csv'),
                write_header=best_metric is None)
        
        if checkpoint_saver is not None:
            save_metric = eval_metrics[eval_metric]
            best_metric, best_epoch = checkpoint_saver.save_checkpoint(epoch, metric=save_metric) 

    if best_metric is not None:
        logger.info('*** Best metric: {0} (epoch {1})'.format(best_metric, best_epoch))

if __name__ == "__main__":
    main()