import arch
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
from ptflops import get_model_complexity_info
import factorize as fr
import baseline as bl
from torchsummary import summary

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

def info(net):
    macs, params = get_model_complexity_info(net, (3, 32, 32), as_strings = True,
                                            print_per_layer_stat = False, verbose = True)

    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))

import res18_arch as res18

if __name__ == '__main__':
    batch_size = 100
    train_loader, val_loader = prepare_loader(batch_size=batch_size)
    criterion, lr, path = nn.CrossEntropyLoss().cuda(), 0.001, "musco.pth" #.cuda()
    epoch = 100

    net = arch.Net() # res18.resnet18()
    net.load_state_dict(torch.load('baseline.pth'))
    net = net.cuda()
    bl.validation(net, val_loader, criterion)
    # summary(net, input_size=(3, 32, 32))

    net = fr.TuckerFactorze(net.cpu()).cuda()
    bl.validation(net, val_loader, criterion)
    # summary(net, input_size=(3, 32, 32))
    optimizer = torch.optim.SGD(net.parameters(), lr = lr, momentum = 0.9)
    # bl.train(net, batch_size, epoch, criterion, optimizer, train_loader, val_loader, path)
    # '''
    step = 2
    for i in range(step):
        net = fr.TuckerMuscoStep(net.cpu(), reduction_rate=0.2).cuda()
        bl.validation(net, val_loader, criterion)
        summary(net, input_size=(3, 32, 32))

        optimizer = torch.optim.SGD(net.parameters(), lr = lr, momentum = 0.9)
        
        # bl.train(net, batch_size, epoch, criterion, optimizer, train_loader, val_loader, path)  
    # '''