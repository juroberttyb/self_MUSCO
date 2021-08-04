import arch
import FacArch as fa
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from ptflops import get_model_complexity_info
import RestoreFac as rf

import time

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def validate(val_loader, net, criterion):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    net.eval()

    for i, (input, target) in enumerate(val_loader):
        
        target = target.cuda()
        input = input.cuda()
        
        with torch.no_grad():
            output = net(input)

        loss = criterion(output, target)

        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    return top1.avg

def train(train_loader, net, criterion, optimizer, epoch):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    net.train()

    for i, (input, target) in enumerate(train_loader):
        input = input.cuda()
        target = target.cuda()

        output = net(input)

        loss = criterion(output, target)

        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        optimizer.zero_grad()
        loss.backward() # retain_graph = True
        optimizer.step()
        
        print_freq = 500
        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, i, len(train_loader), loss=losses, top1=top1, top5=top5))

def data_prepare(traindir, valdir):
    normalize = transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                    std = [0.229, 0.224, 0.225])

    train_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(
        traindir, transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size = 1024, shuffle = True,
        num_workers = 12, pin_memory = True)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size = 1024, shuffle = False,
        num_workers = 12, pin_memory = True)
    
    return train_loader, val_loader

def info(net):
    macs, params = get_model_complexity_info(net, (3, 224, 224), as_strings = True,
                                            print_per_layer_stat = False, verbose = True)

    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))

if __name__ == '__main__':
    net = arch.resnet18(pretrained = True)
    info(net)

    net = fa.resnet18(net).cuda()
    # state_dict = torch.load('55.254ratio0.849.pth', map_location = 'cpu')
    # net.load_state_dict(state_dict)
    info(net)

    # net = rf.resnet18(net).cuda()
    # info(net)

    train_loader, val_loader = data_prepare(traindir = '/home/taoyuan123/imagenet/train', valdir = '/home/taoyuan123/imagenet/val')
    
    criterion = nn.CrossEntropyLoss().cuda()

    best_prec1, lr = 0., 0.001
    optimizer = torch.optim.SGD(net.parameters(), lr = lr, momentum = 0.9)
    # validate(val_loader, net, criterion)

    # print([name for name, param in net.named_parameters()])

    for epoch in range(9999):
        train(train_loader, net, criterion, optimizer, epoch)
        
        prec1 = validate(val_loader, net, criterion)

        if prec1 > best_prec1:  
            best_prec1 = prec1
            torch.save(net.state_dict(), './weak0.5_1/epoch: ' + str(epoch) + ', prec1: ' + str(prec1) + '.pth')
