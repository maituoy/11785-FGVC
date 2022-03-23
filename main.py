from Models import ResNet
from Dataset import *
from prepare import *

import argparse

def parse_argument():

	parser = argparse.ArgumentParser(description='params')
	parser.add_argument('-epochs', dest='epochs', type=int, default=50)
	parser.add_argument('-bs', dest='bs', type=int, default=64)
    parser.add_argument('-lr', dest='lr', type=float, default=0.03)
    parser.add_argument('-wd', dest='wd', type=float, default=1e-3)
    parser.add_argument('-img_size', dest='img_size', type=int, default=256)
    parser.add_argument('-input_size', dest='input_size', type=int, default=224)
    parser.add_argument('-dataset', dest='dataset', type=str, default='cub')

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
		train_dataset = CUB2011(root='./', transform=train_transforms, train=True, extract=True)
		test_dataset = CUB2011(root='./', transform=test_transforms, train=False, extract=True)

	elif dataset == 'dog':
		train_dataset = StandfordDog(root='./', transform=train_transforms, train=True, extract=True)
		test_dataset = StandfordDog(root='./', transform=test_transforms, train=False, extract=True)


	train_loader = DataLoader(train_dataset, batch_size=batch_size, 
	                                              shuffle=True, num_workers=1)
	test_loader = DataLoader(test_dataset, batch_size=batch_size, 
                                              shuffle=False, num_workers=1)

	len_train = len(train_loader)

	model = resnet50(pretrained=True)

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	criterion, optimizer, scheduler, scaler = set_up(model, device, lr, weight_decay, len_train, epochs)

	for epoch in range(epochs):
		train_accuracy, train_loss, learning_rate = train(model, device, batch_size, train_loader, optimizer, criterion, scheduler, scaler)
		print("Epoch {}/{}: Train Acc {:.04f}%, Train Loss {:.04f}, Learning Rate {:.04f}".format(epoch + 1, epochs, train_accuracy, train_loss, learning_rate))

		if not (epoch + 1) % 10 and epoch > 0:
            test_accuracy = evaluate(model, device, batch_size, test_loader, test_dataset)
            print("Test: {:.04f}%".format(test_accuracy))


if __name__ == "__main__":
	main() 



