import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist

from tqdm import tqdm
from warmup_scheduler import GradualWarmupScheduler
from timm.utils import AverageMeter, accuracy
from utils import reduce_tensor


def get_parameter_num(model):
    num_trainable_parameters = 0
    for p in model.parameters():
        num_trainable_parameters += p.numel()
    
    return num_trainable_parameters

def set_up(model, device, lr, weight_decay, len_train, epochs, warmup_epochs=None):

    model = model
    model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(len_train * epochs))
    if warmup_epochs:
        scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs*len_train, after_scheduler=scheduler)

    scaler = torch.cuda.amp.GradScaler()

    return criterion, optimizer, scheduler, scaler


def train(model, device, batch_size, train_loader, optimizer, criterion, scheduler, scaler):

    model.train()

    batch_bar = tqdm(total=len(train_loader), dynamic_ncols=True, leave=False, position=0, desc='Train') 

    num_correct = 0
    total_loss = 0

    for i, (x, y) in enumerate(train_loader):
        optimizer.zero_grad()

        if device.type == 'cuda':
            x = x.cuda()
            y = y.cuda()

        with torch.cuda.amp.autocast():     
            outputs = model(x)
            loss = criterion(outputs, y)

        num_correct += int((torch.argmax(outputs, axis=1) == y).sum())
        total_loss += float(loss)

        batch_bar.set_postfix(
            acc="{:.04f}%".format(100 * num_correct / ((i + 1) * batch_size)),
            loss="{:.04f}".format(float(total_loss / (i + 1))),
            num_correct=num_correct,
            lr="{:.04f}".format(float(optimizer.param_groups[0]['lr'])))
         
        scaler.scale(loss).backward()
        scaler.step(optimizer) 
        scaler.update()

        scheduler.step() 
        batch_bar.update() 

    batch_bar.close() 

    train_accuracy = 100 * num_correct / (len(train_loader) * batch_size)
    train_loss = float(total_loss / len(train_loader))
    learning_rate = float(optimizer.param_groups[0]['lr'])

    return train_accuracy, train_loss, learning_rate

def evaluate(model, device, batch_size, test_loader, test_dataset):

    model.eval()

    batch_bar = tqdm(total=len(test_loader), dynamic_ncols=True, position=0, leave=False, desc='Test')
    num_correct = 0
    for i, (x, y) in enumerate(test_loader):

        if device.type == 'cuda':
            x = x.cuda()
            y = y.cuda()

        with torch.no_grad():
            outputs = model(x)

        num_correct += int((torch.argmax(outputs, axis=1) == y).sum())
        batch_bar.set_postfix(acc="{:.04f}%".format(100 * num_correct / ((i + 1) * batch_size)))

        batch_bar.update()
    
    batch_bar.close()

    test_accuracy = 100 * num_correct / len(test_dataset)

    return test_accuracy

def train_one_epoch(epoch, model, train_loader, optimizer, criterion, scheduler, scaler):
    model.train()
    optimizer.zero_grad()

    num_steps = len(train_loader)
    batch_bar = tqdm(total=num_steps, dynamic_ncols=True, leave=False, position=0, desc='Train') 

    losses_m = AverageMeter()

    for idx, (samples, targets) in enumerate(train_loader):
        samples, targets = samples.cuda(non_blocking=True), targets.cuda(non_blocking=True)

        with torch.cuda.amp.autocast():  
            outputs = model(samples)
            loss = criterion(outputs, targets)

        loss = reduce_tensor(loss)
        losses_m.update(loss.item(), num_steps)
        batch_bar.set_postfix(
            loss="{:.04f}".format(losses_m.avg),
            lr="{:.04f}".format(float(optimizer.param_groups[0]['lr'])))

        scaler.scale(loss).backward()
        scaler.step(optimizer) 
        scaler.update()
        scheduler.step() 
        batch_bar.update() 

        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    batch_bar.close()
    if dist.get_rank() == 0:
        print("Epoch {}: Train Loss {:.04f}, Learning Rate {:.04f}".format(epoch + 1, losses_m.avg, float(optimizer.param_groups[0]['lr'])))

def validate(model, test_loader, criterion):

    losses_m = AverageMeter()
    acc_m = AverageMeter()
    num_steps = len(test_loader)
    for idx, (samples, targets) in enumerate(test_loader):

        samples, targets = samples.cuda(non_blocking=True), targets.cuda(non_blocking=True)

        output = model(samples)

        loss = criterion(output, targets)
        acc = accuracy(output, targets, topk=(1,))[0]

        loss = reduce_tensor(loss)
        acc = reduce_tensor(acc)

        losses_m.update(loss.item(), num_steps)
        acc_m.update(acc.item(), num_steps)
        torch.cuda.empty_cache()
    if dist.get_rank() == 0:
        print("Test Loss: {:.04f}, Test Acc: {:.04f}%".format(losses_m.avg, acc_m.avg))