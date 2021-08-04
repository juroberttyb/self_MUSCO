import numpy as np
import torch
import torch.nn as nn
import tensorly.decomposition as td
import tensorly
import vbmf

class TuckerBlock(nn.Module):
    ''' conv_weight, conv_bias: numpy '''
    def __init__(self, net, padding = 'same', bias = False):
        super(TuckerBlock, self).__init__()

        weight = net.weight.data.numpy()

        # chin, chout = self.complete_rank(weight)
        rank_in, rank_out = self.weakened_rank(weight)

        compress = nn.Conv2d(weight.shape[1], rank_in, kernel_size = 1, bias = False)
        core = nn.Conv2d(rank_in, rank_out, kernel_size = weight.shape[2], padding = padding, bias = False)
        restore = nn.Conv2d(rank_out, weight.shape[0], kernel_size = 1, bias = bias)

        c, [t, s] = td.partial_tucker(weight, modes = [0, 1], rank = [rank_out, rank_in])

        s = np.transpose(s)
        s = np.reshape(s, (s.shape[0], -1, 1, 1))
        compress.weight.data = torch.from_numpy(s.copy())
        
        core.weight.data = torch.from_numpy(c.copy())

        t = np.reshape(t, (t.shape[0], -1, 1, 1))
        restore.weight.data = torch.from_numpy(t.copy())

        if bias:
            restore.bias.data = net.bias.data.numpy()

        self.feature = nn.Sequential(compress, core, restore)

    def exact_rank(self, tensor, tol = None):
        in_rank = int(0.5 * np.linalg.matrix_rank(tensorly.unfold(tensor, 1), tol = tol))
        out_rank = int(0.5 * np.linalg.matrix_rank(tensorly.unfold(tensor, 0), tol = tol))
        # in_rank = vbmf.EVBMF(tensorly.unfold(tensor, 1))[1].shape[0]
        # out_rank = vbmf.EVBMF(tensorly.unfold(tensor, 0))[1].shape[0]
        
        return in_rank, out_rank

    def complete_rank(self, weight):
        return weight.shape[1], weight.shape[0]
    
    def weakened_rank(self, weight, step_size = 0.):
        extreme_in_rank = vbmf.EVBMF(tensorly.unfold(weight, 1))[1].shape[0]
        extreme_out_rank = vbmf.EVBMF(tensorly.unfold(weight, 0))[1].shape[0]
        
        init_in_rank = weight.shape[1]
        init_out_rank = weight.shape[0]
        
        weakened_in_rank = init_in_rank - int(step_size * (init_in_rank - extreme_in_rank))
        weakened_out_rank = init_out_rank - int(step_size * (init_out_rank - extreme_out_rank))
        
        return weakened_in_rank, weakened_out_rank

    def forward(self,x):
        return self.feature(x)

class MuscoBlock(nn.Module):
    ''' conv_weight, conv_bias: numpy '''
    def __init__(self, net, padding = 'same', bias = False):
        super(MuscoBlock, self).__init__()

        block = net.feature

        compress_weight = block[0].weight.data.numpy()
        core_weight = block[1].weight.data.numpy()
        restore_weight = block[2].weight.data.numpy()
        
        rankin, rankout = self.weakened_rank(core_weight)
        kernel_size = core_weight.shape[2]
        input_dim, output_dim = compress_weight.shape[1], restore_weight.shape[0]

        compress = nn.Conv2d(input_dim, rankin, kernel_size = 1, bias = False)
        core = nn.Conv2d(rankin, rankout, kernel_size = kernel_size, padding = padding, bias = False)
        restore = nn.Conv2d(rankout, output_dim, kernel_size = 1, bias = bias)
        
        c, [t, s] = td.partial_tucker(core_weight, modes = [0, 1], rank = [rankout, rankin])

        s = np.transpose(s)
        print(compress_weight.shape)
        compress_weight = np.reshape(compress_weight, (compress_weight.shape[0], -1))
        print(compress_weight.shape)
        print(s.shape)
        s = np.matmul(s, compress_weight)
        s = np.reshape(s, (s.shape[0], -1, 1, 1))
        compress.weight.data = torch.from_numpy(s.copy())
        
        core.weight.data = torch.from_numpy(c.copy())
        print("-----")
        restore_weight = np.reshape(restore_weight, (restore_weight.shape[0], -1))
        print(restore_weight.shape)
        print(t.shape)
        t = np.matmul(restore_weight, t)
        t = np.reshape(t, (t.shape[0], -1, 1, 1))
        print(t.shape)
        print("-----")
        restore.weight.data = torch.from_numpy(t.copy())

        if bias:
            restore.bias.data = net.restore.bias.data.numpy()

        layers = [compress, core, restore]

        self.feature = nn.Sequential(*layers)

    def weakened_rank(self, weight, step_size = 0.2):
        extreme_in_rank = vbmf.EVBMF(tensorly.unfold(weight, 1))[1].shape[0]
        extreme_out_rank = vbmf.EVBMF(tensorly.unfold(weight, 0))[1].shape[0]
        
        init_in_rank = weight.shape[1]
        init_out_rank = weight.shape[0]
        
        weakened_in_rank = init_in_rank - int(step_size * (init_in_rank - extreme_in_rank))
        weakened_out_rank = init_out_rank - int(step_size * (init_out_rank - extreme_out_rank))
        
        return weakened_in_rank, weakened_out_rank
        
    def forward(self,x):
        return self.feature(x)

def factorze(net):
    for e in dir(net):
        layer = getattr(net, e)
        if isinstance(layer, nn.Conv2d):
            print("get conv2d layer " + e)

            shape = layer.weight.data.numpy().shape
            if (shape[1] > 3 and shape[2] > 1 and shape[3] > 1):
                print(e + " is a worthy layer")
                
                setattr(net, e, TuckerBlock(layer))

def MuscoStep(net):
    for e in dir(net):
        layer = getattr(net, e)
        if isinstance(layer, TuckerBlock) or isinstance(layer, MuscoBlock):
            print("get Block " + e)
            
            setattr(net, e, MuscoBlock(layer))
