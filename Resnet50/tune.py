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
import numpy as np

import time


def ratio_counter(net, gate):
    x = abs(torch.cat(gate))

    t = x.size()[0]
    c = 0.
    for i in range(x.size()[0]):
        if x[i] < 0.01:
            c = c + 1.

    return c / t


funnel_t = 10


def FunnelPenalty(gate):
    x = abs(torch.cat(gate))
    x = torch.where(x > 0, x / (funnel_t + x), x / (funnel_t + x))

    return torch.sum(x)

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

    return top1.avg, 

def train(train_loader, net, criterion, optimizer, epoch, gate):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    net.train()

    accumulation_steps = 8
    for i, (input, target) in enumerate(train_loader):
        input = input.cuda()
        target = target.cuda()

        output = net(input)

        sn = 0.002 * FunnelPenalty(gate)
        loss = criterion(output, target) + sn

        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        loss = loss / accumulation_steps

        loss.backward()
        if i % accumulation_steps == 0:
            optimizer.step()
            net.zero_grad()
            losses.update(loss.data.item(), input.size(0))

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
        batch_size = 128, shuffle = True,
        num_workers = 4, pin_memory = True)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size = 128, shuffle = False,
        num_workers = 4, pin_memory = True)
    
    return train_loader, val_loader

def info(net):
    macs, params = get_model_complexity_info(net, (3, 224, 224), as_strings = True,
                                            print_per_layer_stat = False, verbose = True)

    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))

def SetOptimizer(net, lr):
    gate = list(filter(lambda x: 'weight_g' in x[0].split('.') and 'restore' not in x[0].split('.'), net.named_parameters()))
    normal = list(filter(lambda x: not ('weight_g' in x[0].split('.') and 'restore' not in x[0].split('.')), net.named_parameters()))
    gate = [e[1] for e in gate]
    normal = [e[1] for e in normal]
    
    optimizer = torch.optim.SGD([{'params': gate, 'lr': 0.01}, {'params': normal}], lr = lr, momentum = 0.9) # , weight_decay = 1e-5)
    return optimizer

def GatherGate(net):
    gate = []
    for name, param in net.named_parameters():
        tokens = name.split('.')
        if 'weight_g' in tokens and 'restore' not in tokens:
            gate.append(torch.flatten(param))

    return gate

if __name__ == '__main__':
    train_loader, val_loader = data_prepare(traindir = '../imagenet/train', valdir = '../imagenet/val')
    # train_loader, val_loader = data_prepare(traindir = '/home/jurobert/Desktop/Gate/imagenet', valdir = '/home/jurobert/Desktop/Gate/imagenet')

    criterion = nn.CrossEntropyLoss().cuda()
    
    net = arch.resnet50(pretrained = True)
    info(net)
    net = fa.resnet50(net).cuda()
    state_dict = torch.load('76.922.pth')
    net.load_state_dict(state_dict)
    info(net)

    optimizer = SetOptimizer(net, lr = 0.001)

    gate = GatherGate(net)
    
    epoch = 0
    while True:
        if epoch <= 100:
            funnel_t = 10 + (-0.09999) * epoch
        
        train(train_loader, net, criterion, optimizer, epoch, gate)
        
        ratio = ratio_counter(net, gate)
        
        prec1 = validate(val_loader, net, criterion)

        torch.save(net.state_dict(), './model/epoch: ' + str(epoch) + ', acc: ' + str(prec1) + ', ratio: ' + str(ratio) + '.pth')
        
        epoch = epoch + 1




