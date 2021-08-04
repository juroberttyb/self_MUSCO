import arch
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
from ptflops import get_model_complexity_info
import factorize as fr
import baseline as bl

import time

def prepare_loader(batch_size):
    train_transform = transforms.Compose(
                [transforms.ToTensor(),
                 transforms.RandomHorizontalFlip(),
                 transforms.Normalize((0.45, 0.45, 0.45), (0.225, 0.225, 0.225))])

    val_transform = transforms.Compose(
                [transforms.ToTensor(),
                 transforms.Normalize((0.45, 0.45, 0.45), (0.225, 0.225, 0.225))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=train_transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=2)

    val_set = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=val_transform)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size,
                                            shuffle=False, num_workers=2)

    return train_loader, val_loader

def plot_loss(train_loss, val_loss):

    plt.plot(train_loss, 'b')
    plt.plot(val_loss, 'r')
    plt.title('loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig('loss.png')

def validation(val_loader, criterion):
    correct = 0
    total = 0
    val_loss = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in val_loader:
            images, labels = data
            images, labels = images, labels
            # calculate outputs by running images through the network
            outputs = net(images)
            val_loss = val_loss + criterion(outputs, labels)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))

    val_loss = val_loss / 10000

    return val_loss

def info(net):
    macs, params = get_model_complexity_info(net, (3, 32, 32), as_strings = True,
                                            print_per_layer_stat = False, verbose = True)

    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))

if __name__ == '__main__':
    train_loader, val_loader = prepare_loader(batch_size=100)
    criterion, lr = nn.CrossEntropyLoss(), 0.001 #.cuda()

    net = arch.Net()
    net.load_state_dict(torch.load('baseline.pth'))
    validation(val_loader, criterion)
    info(net)

    fr.factorze(net)
    validation(val_loader, criterion)
    info(net)
    train()

    step = 3
    for i in range(step):
        fr.MuscoStep(net)
        validation(val_loader, criterion)
        info(net)

        best_prec1 = 0.
        optimizer = torch.optim.SGD(net.parameters(), lr = lr, momentum = 0.9)
        # validate(val_loader, net, criterion)

        # print([name for name, param in net.named_parameters()])

        epoch = 1
        while True:
            train(train_loader, net, criterion, optimizer, epoch)
            
            prec1 = validate(val_loader, net, criterion)

            if prec1 > best_prec1:  
                best_prec1 = prec1
                torch.save(net.state_dict(), 'musco [%d %.2f].pth' % (epoch, prec1))

            epoch = epoch + 1
        '''