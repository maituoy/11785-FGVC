from Models.ResNet import *
from Dataset import *
from prepare import *
from utils import *

import argparse
from torch.nn.parallel import DistributedDataParallel as DDP

def parse_argument():

    parser = argparse.ArgumentParser(description='params')
    parser.add_argument('-epochs', dest='epochs', type=int, default=100)
    parser.add_argument('-bs', dest='bs', type=int, default=256)
    parser.add_argument('-lr', dest='lr', type=float, default=0.03)
    parser.add_argument('-wd', dest='wd', type=float, default=1e-3)
    parser.add_argument('-img_size', dest='img_size', type=int, default=256)
    parser.add_argument('-input_size', dest='input_size', type=int, default=224)
    parser.add_argument('-dataset', dest='dataset', type=str, default='cub')
    parser.add_argument('-ngpu_used', dest='ngpu_used', type=int, default=1)

    return parser.parse_args()


def main():

    args = parse_argument()
    epochs = args.epochs
    batch_size = args.bs
    lr = args.lr
    weight_decay = args.wd
    img_size = args.img_size
    input_size = args.input_size
    dataset = args.dataset
    ngpu_used = args.ngpu_used
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ddp = 0
    if ngpu_used > 1:
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        n_gpus = torch.cuda.device_count()

        assert n_gpus >= ngpu_used, "GPU is not enough"

        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
        torch.distributed.barrier()
        ddp += 1

    train_transforms = transforms.Compose([transforms.Resize((img_size, img_size), Image.BILINEAR),
                                       transforms.RandomCrop((input_size, input_size)),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

    test_transforms = transforms.Compose([transforms.Resize((img_size, img_size), Image.BILINEAR),
                                           transforms.CenterCrop((input_size, input_size)),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

    if dataset == 'cub':
        train_dataset = CUB2011(root='/media/Bootes/dl_fgvc/', transform=train_transforms, train=True, extract=False)
        test_dataset = CUB2011(root='/media/Bootes/dl_fgvc/', transform=test_transforms, train=False, extract=False)

    elif dataset == 'dog':
        train_dataset = StandfordDog(root='/media/Bootes/dl_fgvc/', transform=train_transforms, train=True, extract=False)
        test_dataset = StandfordDog(root='/media/Bootes/dl_fgvc/', transform=test_transforms, train=False, extract=False)

    if ddp:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
        train_loader = DataLoader(train_dataset, batch_size=batch_size//ngpu_used, 
                                                    shuffle=False, num_workers=8, sampler=train_sampler)
        test_loader = DataLoader(test_dataset, batch_size=batch_size//ngpu_used, 
                                                shuffle=False, num_workers=8, sampler=test_sampler)
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=1)

    len_train = len(train_loader)

    model = resnet50(pretrained=True)
    model.fc = nn.Linear(2048, 200)
    model = model.cuda()

    criterion, optimizer, scheduler, scaler = set_up(model, device, lr, weight_decay, len_train, epochs, warmup_epochs=5)

    if ddp:
        model = DDP(model, device_ids=[local_rank])

    for epoch in range(epochs):
        if ddp:
            train_one_epoch(epoch, model, train_loader, optimizer, criterion, scheduler, scaler)
            validate(model, test_loader, criterion)
        else:
            train_accuracy, train_loss, learning_rate = train(model, device, batch_size, train_loader, optimizer, criterion, scheduler, scaler)
            print("Epoch {}/{}: Train Acc {:.04f}%, Train Loss {:.04f}, Learning Rate {:.04f}".format(epoch + 1, epochs, train_accuracy, train_loss, learning_rate))

            if not (epoch + 1) % 10 and epoch > 0:
                test_accuracy = evaluate(model, device, batch_size, test_loader, test_dataset)
                print("Test: {:.04f}%".format(test_accuracy))


if __name__ == "__main__":
    main() 



