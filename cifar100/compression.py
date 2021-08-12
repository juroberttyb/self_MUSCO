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

def resnet_MuscoStep(net, reduction_rate):
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
                            setattr(block, e2, fr.MuscoTucker(layer, reduction_rate = reduction_rate))
                        elif isinstance(layer, fr.SVDBlock) or isinstance(layer, fr.MuscoSVD):
                            setattr(block, e2, fr.MuscoSVD(layer, reduction_rate = reduction_rate))

    return net

def Res50_MUSCO_approach():
    epoch = 10
    train_loader, val_loader = bl.prepare_loader(batch_size=bl.batch_size)
    criterion, lr, path = nn.CrossEntropyLoss().cuda(), 0.001, "musco.pth" #.cuda()

    net = res.resnet50() # arch.Net() # res18.resnet18()
    net.load_state_dict(torch.load('baseline.pth', map_location='cpu'))
    net = net.cuda()
    summary(net, input_size=(3, 32, 32))
    # print(net)
    bl.validation(net, val_loader, criterion)
    print('load success')

    net = resnet_Tucker_factorize(net.cpu()).cuda() # fr.TuckerFactorze(net.cpu()).cuda()
    summary(net, input_size=(3, 32, 32))
    # print(net)
    bl.validation(net, val_loader, criterion)
    optimizer = torch.optim.SGD(net.parameters(), lr = lr, momentum = 0.9)
    print('factorize success')

    # bl.train(net, bl.batch_size, epoch, criterion, optimizer, train_loader, val_loader, path)

    # '''
    step = 20
    for i in range(step):
        net = resnet_MuscoStep(net.cpu(), reduction_rate=0.1).cuda()
        summary(net, input_size=(3, 32, 32))
        # print(net)
        bl.validation(net, val_loader, criterion)
        print("MUSCO step %d" % i)

        optimizer = torch.optim.SGD(net.parameters(), lr = lr, momentum = 0.9)
        
        bl.train(net, bl.batch_size, epoch, criterion, optimizer, train_loader, val_loader, path)  
    # '''

def resnet_CP_factorize(net, rank):
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
                                setattr(block, e2, fr.CPBlock(layer, rank=rank))
                            elif (layer.kernel_size == (1,1)):
                                # print("svd decomposing " + str(layer))
                                # setattr(block, e2, fr.SVDBlock(layer))
                                ...
                            else:
                                print("nothing matched")

    return net

def Res50_CPTPM_approach(probe_rank, approx_rank):
    epoch = 10
    train_loader, val_loader = bl.prepare_loader(batch_size=bl.batch_size)
    criterion, lr, path = nn.CrossEntropyLoss().cuda(), 0.001, "cptpm.pth" #.cuda()

    net = res.resnet50() # arch.Net() # res18.resnet18()
    net.load_state_dict(torch.load('baseline.pth', map_location='cpu'))
    net = net.cuda()
    # summary(net, input_size=(3, 32, 32))
    # print(net)
    # bl.validation(net, val_loader, criterion)
    print('load success')

    net = resnet_CP_factorize(net.cpu(), probe_rank).cuda()
    # summary(net, input_size=(3, 32, 32))
    # print(net)
    bl.validation(net, val_loader, criterion)
    optimizer = torch.optim.SGD(net.parameters(), lr = lr, momentum = 0.9)
    print('factorize success')

if __name__ == '__main__':
    Res50_MUSCO_approach()