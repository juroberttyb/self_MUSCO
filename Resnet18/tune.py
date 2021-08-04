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

c = None
m = None
gate = None
threshold = 0.01
magnitude = 0.01


def ratio_counter(net):
    global threshold, gate
    x = abs(torch.cat(gate))

    count = 0.
    for i in range(x.size()[0]):
        if x[i] < threshold:
            count = count + 1

    return count / x.size()[0]

def FunnelPenalty():
    global c, m, gate
    
    x = abs(torch.cat(gate))
    
    # '''
    # x = torch.where(x > 0, x / (c + x), x / (c + x))
    for i in range(x.size()[0]):
        x[i] = m[i] * x[i] / (c[i] + x[i])
    # '''
        
    '''
    x1 = torch.mul(x, m)
    x2 = torch.add(x, c)
    x = torch.div(x1, x2)
    '''

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
    print(loss.item())

    return top1.avg

def train(train_loader, net, criterion, optimizer, epoch):
    global c, threshold, magnitude, gate
    
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    net.train()

    # accumulation_steps = 1
    for i, (input, target) in enumerate(train_loader):
        input = input.cuda()
        target = target.cuda()

        output = net(input)

        sn = magnitude * FunnelPenalty()
        loss = criterion(output, target) + sn
        # loss = criterion(output, target) + FunnelPenalty()

        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        # loss = loss / accumulation_steps

        loss.backward()
        # if i % accumulation_steps == 0:
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
        batch_size = 900, shuffle = True,
        num_workers = 8, pin_memory = True)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size = 900, shuffle = False,
        num_workers = 8, pin_memory = True)
    
    return train_loader, val_loader

def info(net):
    macs, params = get_model_complexity_info(net, (3, 224, 224), as_strings = True,
                                            print_per_layer_stat = False, verbose = True)

    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))

# +
def SetOptimizer(net, lr):
    gate = list(filter(lambda x: 'weight_g' in x[0].split('.') and 'restore' not in x[0].split('.'), net.named_parameters()))
    normal = list(filter(lambda x: not ('weight_g' in x[0].split('.') and 'restore' not in x[0].split('.')), net.named_parameters()))
    gate = [e[1] for e in gate]
    normal = [e[1] for e in normal]
    
    optimizer = torch.optim.SGD([{'params': gate, 'lr': 0.01}, {'params': normal}], lr = lr, momentum = 0.0) # , weight_decay = 1e-5)
    return optimizer

def GatherGate(net):
    gate = []
    for name, param in net.named_parameters():
        tokens = name.split('.')
        if 'weight_g' in tokens and 'restore' not in tokens:
            gate.append(torch.flatten(param))

    return gate

def init():
    global c, m, gate
    
    x = abs(torch.cat(gate))
    for i in range(x.size()[0]):
        c[i] = 0.01 * x[i] # funnel5: 0.000001, funnel4: 0.00001 funnel3: 0.0001 funnel2: 0.001, funnel mag1 mag2: 0.01
        m[i] = x[i]


# -

if __name__ == '__main__':
    net = arch.resnet18(pretrained = True)
    info(net)
    net = fa.resnet18(net).cuda()
    state_dict = torch.load('70.734.pth')
    net.load_state_dict(state_dict)
    info(net)

    train_loader, val_loader = data_prepare(traindir = '../imagenet/train', valdir = '../imagenet/val')
    
    criterion = nn.CrossEntropyLoss().cuda()

    optimizer = SetOptimizer(net, lr = 0.001)
    
    gate = GatherGate(net)
    # c = 0.1 * torch.clone(torch.cat(gate))
    # m = torch.clone(torch.cat(gate))
    c = np.zeros(torch.cat(gate).size())
    m = np.zeros(torch.cat(gate).size())
    init()
    
    epoch = 1
    while True:
        train(train_loader, net, criterion, optimizer, epoch)
        
        prec1 = validate(val_loader, net, criterion)

        ratio = ratio_counter(net)
        
        torch.save(net.state_dict(), './model/e' + str(epoch) + ' a' + str(prec1) + ' r' + str(ratio) + '.pth')
        
        epoch = epoch + 1




