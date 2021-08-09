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

def info(net):
    macs, params = get_model_complexity_info(net, (3, 32, 32), as_strings = True,
                                            print_per_layer_stat = False, verbose = True)

    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))

import res18_arch as res18

def resnet18_factorize(net):
    for e1 in dir(net):
        sequential = getattr(net, e1) # layer1, layer2...

        if isinstance(sequential, nn.modules.container.Sequential):
            print('found sequential ' + str(sequential))
            for block in sequential: # block
                
                if isinstance(block, res18.BasicBlock):
                    for e2 in dir(block): # layer
                        layer = getattr(block, e2)

                        if isinstance(layer, nn.Conv2d):
                            if (layer.in_channels > 3 and layer.kernel_size[0] > 1 and layer.kernel_size[1] > 1):
                                print("decomposing " + str(layer))
                                setattr(block, e2, fr.TuckerBlock(layer))

    return net

if __name__ == '__main__':
    batch_size, epoch = 100, 100
    train_loader, val_loader = bl.prepare_loader(batch_size=batch_size)
    criterion, lr, path = nn.CrossEntropyLoss().cuda(), 0.001, "musco.pth" #.cuda()

    net = res18.resnet18() # arch.Net() # res18.resnet18()
    # net.load_state_dict(torch.load('baseline.pth'))
    net = net.cuda()
    bl.validation(net, val_loader, criterion)
    # summary(net, input_size=(3, 32, 32))

    net = resnet18_factorize(net.cpu()).cuda() # fr.TuckerFactorze(net.cpu()).cuda()
    bl.validation(net, val_loader, criterion)
    # summary(net, input_size=(3, 32, 32))
    optimizer = torch.optim.SGD(net.parameters(), lr = lr, momentum = 0.9)
    # bl.train(net, batch_size, epoch, criterion, optimizer, train_loader, val_loader, path)
    
    # print(net.layer1)
    '''
    step = 2
    for i in range(step):
        net = fr.TuckerMuscoStep(net.cpu(), reduction_rate=0.2).cuda()
        bl.validation(net, val_loader, criterion)
        summary(net, input_size=(3, 32, 32))

        optimizer = torch.optim.SGD(net.parameters(), lr = lr, momentum = 0.9)
        
        # bl.train(net, batch_size, epoch, criterion, optimizer, train_loader, val_loader, path)  
    '''