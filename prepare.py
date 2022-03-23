import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm

def get_parameter_num(model):
    num_trainable_parameters = 0
    for p in model.parameters():
        num_trainable_parameters += p.numel()
    
    return num_trainable_parameters

def set_up(model, device, lr, weight_decay, len_train, epochs):

    model = model
    model.to(device)

    criterion = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(len_train * epochs))

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

