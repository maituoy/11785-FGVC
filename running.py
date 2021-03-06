import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist

from tqdm import tqdm
from warmup_scheduler import GradualWarmupScheduler
from timm.utils import AverageMeter, accuracy
from utils import reduce_tensor
import time
from collections import OrderedDict
import logging

logger = logging.getLogger(__name__)

def setup4training(model, config, resume_epoch=None, optimizer_state=None, scaler_state=None):

    parameters = model.parameters()

    if config.optim.name == 'sgd':
        optimizer = torch.optim.SGD(parameters, lr=config.optim.lr, momentum=0.9, weight_decay=config.optim.weight_decay)
    elif config.optim.name == 'adamw':
        optimizer = torch.optim.AdamW(parameters, lr=config.optim.lr, weight_decay=config.optim.weight_decay)
    else:
        raise NotImplementedError

    scaler = torch.cuda.amp.GradScaler(enabled=config.use_amp)

    if resume_epoch is None:
        if config.optim.sched == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 
                                                            T_max=(config.data.batches * config.optim.epochs)
                                                            )
        else:
            raise NotImplementedError
                    
        if config.optim.warmup_epochs > 0:
            scheduler = GradualWarmupScheduler(optimizer, 
                                            multiplier=1, 
                                            total_epoch=config.optim.warmup_epochs * config.data.batches, 
                                            after_scheduler=scheduler)
    else:
        optimizer.load_state_dict(optimizer_state)
        scaler = scaler_state
        if resume_epoch > config.optim.warmup_epochs:
            if config.optim.sched == 'cosine':
                scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 
                                                            T_max=(config.data.batches * (config.optim.epochs - (resume_epoch + 1 - config.optim.warmup_epochs)))
                                                            )
            else:
                raise NotImplementedError
        else:
            if config.optim.sched == 'cosine':
                scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 
                                                            T_max=(config.data.batches * config.optim.epochs)
                                                            )
            else:
                raise NotImplementedError
            
            scheduler = GradualWarmupScheduler(optimizer, 
                                            multiplier=1, 
                                            total_epoch=(config.optim.warmup_epochs - resume_epoch - 1) * config.data.batches, 
                                            after_scheduler=scheduler)
    return optimizer, scheduler, scaler


def train_one_epoch(epoch, model, train_loader, optimizer, criterion, scheduler, scaler, config, mixup_fn=None,model_ema=None):

    model.train()
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    losses_m = AverageMeter()
    
    end = time.time()
    last_idx = len(train_loader) - 1
    num_updates = epoch * len(train_loader)

    for idx, (samples, targets) in enumerate(train_loader):
        last_batch = last_idx == idx
        data_time_m.update(time.time() - end)
        samples, targets = samples.cuda(), targets.cuda()

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast(enabled=True):  
            outputs = model(samples)
            loss = criterion(outputs, targets)
        
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        scaler.step(optimizer) 
        scaler.update()
        torch.cuda.synchronize()
        
        if model_ema is not None:
            model_ema.update(model)
          
        if not config.distributed:
            losses_m.update(loss.item(), samples.size(0))
        # else:
        #     reduced_loss = reduce_tensor(loss)
        #     losses_m.update(reduced_loss.item(), samples.size(0))

        num_updates += 1
        batch_time_m.update(time.time() - end)

        if last_batch or idx % config.log_interval == 0:
            lr = optimizer.param_groups[0]['lr']

            if config.distributed:
                reduced_loss = reduce_tensor(loss.data, config.world_size)
                losses_m.update(reduced_loss.item(), samples.size(0))

            if config.local_rank == 0:
                logger.info(
                    'Train: {} [{:>4d}/{} ({:>3.0f}%)]  '
                    'Loss: {loss.val:>9.6f} ({loss.avg:>6.4f})  '
                    'Time: {batch_time.val:.3f}s, {rate:>7.2f}/s  '
                    '({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s)  '
                    'LR: {lr:.3e}  '
                    'Data: {data_time.val:.3f} ({data_time.avg:.3f})'.format(
                        epoch,
                        idx, len(train_loader),
                        100. * idx / last_idx,
                        loss=losses_m,
                        batch_time=batch_time_m,
                        rate=samples.size(0) * config.world_size / batch_time_m.val,
                        rate_avg=samples.size(0) * config.world_size / batch_time_m.avg,
                        lr=lr,
                        data_time=data_time_m))

        scheduler.step()

        end = time.time()
    
    return OrderedDict([('train_loss', losses_m.avg)])
        

def val_one_epoch(model, test_loader, config):

    batch_time_m = AverageMeter()
    losses_m = AverageMeter()
    top1_m = AverageMeter()
    top5_m = AverageMeter()
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()

    num_steps = len(test_loader)
    end = time.time()
    last_idx = num_steps - 1
    
    with torch.no_grad():
        for idx, (samples, targets) in enumerate(test_loader):
            last_batch = last_idx == idx
            samples, targets = samples.cuda(), targets.cuda()

            output = model(samples)

            loss = criterion(output, targets)
            acc1, acc5 = accuracy(output, targets, topk=(1,5))

            if config.distributed:
                reduced_loss = reduce_tensor(loss.data, config.world_size)
                
                acc1 = reduce_tensor(acc1, config.world_size)
                acc2 = reduce_tensor(acc5, config.world_size)

            else:
                reduced_loss = loss.data

            torch.cuda.synchronize()
            losses_m.update(reduced_loss.item(), samples.size(0))
            top1_m.update(acc1.item(), output.size(0))
            top5_m.update(acc5.item(), output.size(0))

            batch_time_m.update(time.time() - end)
            end = time.time()

            if config.local_rank == 0 and (last_batch or idx % config.log_interval == 0):
                log_name = 'Test'
                logger.info(
                    '{0}: [{1:>4d}/{2}]  '
                    'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})  '
                    'Loss: {loss.val:>7.4f} ({loss.avg:>6.4f})  '
                    'Acc@1: {top1.val:>7.4f} ({top1.avg:>7.4f})  '
                    'Acc@5: {top5.val:>7.4f} ({top5.avg:>7.4f})'.format(
                        log_name, idx, last_idx, batch_time=batch_time_m,
                        loss=losses_m, top1=top1_m, top5=top5_m))
            

    metrics = OrderedDict([('val_loss', losses_m.avg), 
                           ('val_top1', top1_m.avg),
                           ('val_top5', top5_m.avg)])
    return metrics
