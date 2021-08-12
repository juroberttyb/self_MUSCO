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
    macs, params = get_model_complexity_info(net, (3, 224, 224), as_strings = True,
                                            print_per_layer_stat = False, verbose = True)

    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))

import res_arch as res

def resnet_Tucker_factorize(net):
    for e1 in dir(net):
        sequential = getattr(net, e1) # layer1, layer2...

        if isinstance(sequential, nn.modules.container.Sequential):
            # print('found sequential ' + str(sequential))
            for block in sequential: # block
                
                if isinstance(block, res.Bottleneck):
                    for e2 in dir(block): # layer
                        layer = getattr(block, e2)

                        if isinstance(layer, nn.Conv2d):
                            if (layer.in_channels > 3 and layer.kernel_size != (1,1)):
                                # print("tucker decomposing " + str(layer))
                                setattr(block, e2, fr.TuckerBlock(layer))
                            elif (layer.kernel_size == (1,1)):
                                # print("svd decomposing " + str(layer))
                                setattr(block, e2, fr.SVDBlock(layer))
                            else:
                                print("nothing matched")

    return net

def resnet_MuscoStep(net, table, reduction_rate):
    i = 0
    for e1 in dir(net):
        sequential = getattr(net, e1) # layer1, layer2...

        if isinstance(sequential, nn.modules.container.Sequential):
            # print('found sequential ' + str(sequential))
            for block in sequential: # block
                
                if isinstance(block, res.Bottleneck):
                    for e2 in dir(block): # layer
                        layer = getattr(block, e2)

                        if isinstance(layer, fr.TuckerBlock) or isinstance(layer, fr.MuscoTucker):
                            # print("MUSCO decomposing " + str(layer))
                            if table[i]:
                                setattr(block, e2, fr.MuscoTucker(layer, reduction_rate = reduction_rate))
                            i = i + 1
                        elif isinstance(layer, fr.SVDBlock) or isinstance(layer, fr.MuscoSVD):
                            if table[i]:
                                setattr(block, e2, fr.MuscoSVD(layer, reduction_rate = reduction_rate))
                            i = i + 1

    return net

def sensitive_check(net, reduction_rate, val_loader, criterion):
    val_loss = []
    for e1 in dir(net):
        sequential = getattr(net, e1) # layer1, layer2...

        if isinstance(sequential, nn.modules.container.Sequential):
            # print('found sequential ' + str(sequential))
            for block in sequential: # block
                
                if isinstance(block, res.Bottleneck):
                    for e2 in dir(block): # layer
                        layer = getattr(block, e2)
                        temp = layer

                        if isinstance(layer, fr.TuckerBlock) or isinstance(layer, fr.MuscoTucker):
                            # print("MUSCO decomposing " + str(layer))
                            setattr(block, e2, fr.MuscoTucker(layer, reduction_rate = reduction_rate))
                            val_loss.append(bl.validation(net.cuda(), val_loader, criterion, fast=True))
                            net.cpu()
                            setattr(block, e2, temp)
                        elif isinstance(layer, fr.SVDBlock) or isinstance(layer, fr.MuscoSVD):
                            setattr(block, e2, fr.MuscoSVD(layer, reduction_rate = reduction_rate))
                            val_loss.append(bl.validation(net.cuda(), val_loader, criterion, fast=True))
                            net.cpu()
                            setattr(block, e2, temp)
                        
    temp = val_loss.copy()
    temp.sort()
    mid = temp[int(len(temp)/2)]
    for i in range(len(val_loss)):
        if val_loss[i] > mid:
            val_loss[i] = True
        else:
            val_loss[i] = False

    return val_loss

def Res50_MUSCO_approach():
    epoch = 20
    train_loader, val_loader = bl.prepare_loader(batch_size=bl.batch_size)
    criterion, lr, path = nn.CrossEntropyLoss().cuda(), 0.001, "musco.pth" #.cuda()

    net = res.resnet50() # arch.Net() # res18.resnet18()
    net.load_state_dict(torch.load('baseline.pth', map_location='cpu'))
    # net = net.cuda()
    info(net)
    # print(net)
    # bl.validation(net.cuda(), val_loader, criterion)
    print('load success')

    net = resnet_Tucker_factorize(net).cuda()
    info(net)
    # print(net)
    # bl.validation(net, val_loader, criterion)
    optimizer = torch.optim.SGD(net.parameters(), lr = lr, momentum = 0.9)
    print('factorize success')

    # bl.train(net, bl.batch_size, epoch, criterion, optimizer, train_loader, val_loader, path)

    # '''
    step, reduction_rate = 50, 0.1
    for i in range(step):
        print("doing sensitive check...")
        # bl.validation(net, val_loader, criterion)
        table = sensitive_check(net.cpu(), reduction_rate, val_loader, criterion)
        print("doing sensitive done")
        # bl.validation(net.cuda(), val_loader, criterion)
        net = resnet_MuscoStep(net.cpu(), table, reduction_rate).cuda()
        info(net)
        # print(net)
        print("MUSCO step %d" % i)
        bl.validation(net, val_loader, criterion)

        optimizer = torch.optim.SGD(net.parameters(), lr = lr, momentum = 0.9)
        
        bl.train(net, bl.batch_size, epoch, criterion, optimizer, train_loader, val_loader, path)  
    # '''

if __name__ == '__main__':
    Res50_MUSCO_approach()