from math import fabs
from os import stat
import numpy as np
import torch
import torch.nn as nn
import tensorly.decomposition as td
import tensorly
import vbmf

# traditional SVD approach without lower rank
class SVDBlock(nn.Module):
    ''' conv_weight, conv_bias: numpy '''
    def __init__(self, layer, rank=None):
        super(SVDBlock, self).__init__()
        
        weight = layer.weight.data.numpy()
        
        self.feature = SVDBlock.SVDfactorize(weight, rank, layer.bias)

    @staticmethod
    def SVDfactorize(weight, rank, bias):
        u, v = SVDBlock.TorchFormSVD(weight, rank)

        element = nn.Conv2d(u.shape[1], u.shape[0], kernel_size = 1, bias = False)
        if bias == None:
            restore = nn.Conv2d(v.shape[1], v.shape[0], kernel_size = 1, bias = False)
        else:
            restore = nn.Conv2d(v.shape[1], v.shape[0], kernel_size = 1, bias = True)
            restore.bias.data = bias.data

        element.weight.data = torch.from_numpy(u.copy())
        restore.weight.data = torch.from_numpy(v.copy())

        return nn.Sequential(element, restore)

    @staticmethod
    def TorchFormSVD(filters, rank, with_s=False):
        filters = np.transpose(filters.reshape((filters.shape[0], -1)))

        if with_s:
            u, s, v = SVDBlock.SVD(filters, with_s=with_s, rank=rank)
        else:
            u, v = SVDBlock.SVD(filters, with_s=with_s, rank=rank)
        
        u = u.transpose()
        u = u.reshape((*u.shape, 1, 1))
        
        v = v.transpose()
        v = v.reshape((*v.shape, 1, 1))

        # s = np.sqrt(s)
        if with_s:
            s = s.reshape(s.shape[0], 1, 1, 1)
            return u, s, v
        return u, v

    @staticmethod
    def SVD(weight, with_s, rank=None):
        u, s, v = np.linalg.svd(weight, full_matrices=False)
        # full_matrices=False -> the shapes are (..., M, K) and (..., K, N), respectively, where K = min(M, N)
        
        '''
        print('--- before')
        print(u.shape)
        print(s.shape)
        print(v.shape)
        print('---')
        '''
        
        if rank != None:
            s = s[:rank]

        for i in range(s.shape[0]): # clear singular values == 0
            if s[i] <= 0.:
                s = s[:i]
                break

        u = u[:, :s.shape[0]]
        v = v[:s.shape[0], :]

        '''
        print('--- after')
        print(u.shape)
        print(s.shape)
        print(v.shape)
        print('---')
        '''
        
        if with_s:
            return u, s, v

        s = np.sqrt(s)

        for i in range(s.shape[0]):
            u[:, i] = u[:, i] * s[i]

        for i in range(s.shape[0]):
            v[i, :] = v[i, :] * s[i]
        
        return u, v

    def forward(self, x):    
        return self.feature(x)

# Dense layer SVD approach, not yet implemented but described in 
# "CP-decomposition with Tensor Power Method for Convolutional Neural Networks Compression"
class DenseBlock(nn.Module):
    ''' conv_weight, conv_bias: numpy '''
    def __init__(self, layer, rank=None):
        super(DenseBlock, self).__init__()

        weight = layer.weight.data.numpy()
        
        self.feature = SVDBlock.SVDfactorize(weight, rank, layer.bias)

    def forward(self, x):    
        return self.feature(x)

# traditional Tucker approach without lower rank
class TuckerBlock(nn.Module):
    ''' conv_weight, conv_bias: numpy '''
    def __init__(self, layer, rank_mode = 'exact'):
        super(TuckerBlock, self).__init__()

        weight = layer.weight.data.numpy()
        rank_in, rank_out = TuckerBlock.rank_select(weight, rank_mode)

        compress = nn.Conv2d(layer.in_channels, rank_in, kernel_size = 1, bias = False)
        core = nn.Conv2d(rank_in, rank_out, kernel_size = layer.kernel_size, 
                         stride = layer.stride, padding = layer.padding, 
                         groups = layer.groups, dilation = layer.dilation, bias = False)
        if layer.bias != None:
            restore = nn.Conv2d(rank_out, layer.out_channels, kernel_size = 1, bias = True)
            restore.bias.data = layer.bias.data
        else:
            restore = nn.Conv2d(rank_out, layer.out_channels, kernel_size = 1, bias = False)

        c, [t, s] = td.partial_tucker(weight, modes = [0, 1], rank = [rank_out, rank_in])

        s = np.transpose(s)
        s = np.reshape(s, (s.shape[0], -1, 1, 1))
        compress.weight.data = torch.from_numpy(s.copy())
        
        core.weight.data = torch.from_numpy(c.copy())

        t = np.reshape(t, (t.shape[0], -1, 1, 1))
        restore.weight.data = torch.from_numpy(t.copy())

        self.feature = nn.Sequential(compress, core, restore)

    @staticmethod
    def rank_select(weight, rank_mode):
        if rank_mode == 'exact':
            return TuckerBlock.exact_rank(weight)
        elif rank_mode == 'complete':
            return TuckerBlock.complete_rank(weight)
        else:
            return TuckerBlock.lower_rank(weight)

    @staticmethod
    def exact_rank(tensor, tol = None):
        in_rank = np.linalg.matrix_rank(tensorly.unfold(tensor, 1), tol = tol)
        out_rank = np.linalg.matrix_rank(tensorly.unfold(tensor, 0), tol = tol)
        
        return in_rank, out_rank
    
    @staticmethod
    def complete_rank(weight):
        return weight.shape[1], weight.shape[0]
    
    @staticmethod
    def lower_rank(weight):
        extreme_in_rank = vbmf.EVBMF(tensorly.unfold(weight, 1))[1].shape[0]
        extreme_out_rank = vbmf.EVBMF(tensorly.unfold(weight, 0))[1].shape[0]

        return extreme_in_rank, extreme_out_rank

    def forward(self,x):
        return self.feature(x)

# self-modified SVD version of "Automated Multi-Stage Compression of Neural Networks"
class MuscoSVD(nn.Module):
    ''' conv_weight, conv_bias: numpy '''
    def __init__(self, block, reduction_rate):
        super(MuscoSVD, self).__init__()

        element = block.feature[0].weight.data.numpy()
        restore = block.feature[1].weight.data.numpy()

        weight = self.Reform(element, restore)
        rank = self.weakened_rank(weight, reduction_rate)

        weight = weight.transpose()
        weight = weight.reshape((*weight.shape, 1, 1))

        self.feature = SVDBlock.SVDfactorize(weight, rank, block.feature[1].bias)

    def Reform(self, u, v):

        u = u.reshape((u.shape[0], -1))
        u = u.transpose()
        
        v = v.reshape((v.shape[0], -1))
        v = v.transpose()

        return np.matmul(u, v)

    def weakened_rank(self, weight, reduction_rate):
        '''
        extreme_rank = vbmf.EVBMF(tensorly.unfold(weight, 1))[1].shape[0]
        
        init_rank = np.linalg.matrix_rank(weight, tol = None)
        
        weakened_rank = init_rank - int(reduction_rate * (init_rank - extreme_rank))
        '''

        weakened_rank = int(np.linalg.matrix_rank(weight, tol = None) * (1. - reduction_rate))
        
        return weakened_rank

    def forward(self, x):
        return self.feature(x)

# self-modified Tucker version of "Automated Multi-Stage Compression of Neural Networks"
class MuscoTucker(nn.Module):
    ''' conv_weight, conv_bias: numpy '''
    def __init__(self, block, reduction_rate):
        super(MuscoTucker, self).__init__()

        compress_weight = block.feature[0].weight.data.numpy()
        core_weight = block.feature[1].weight.data.numpy()
        restore_weight = block.feature[2].weight.data.numpy()
        
        rankin, rankout = self.weakened_rank(core_weight, reduction_rate)

        compress = nn.Conv2d(block.feature[0].in_channels, rankin, kernel_size = 1, bias = False)
        core = nn.Conv2d(rankin, rankout, kernel_size = block.feature[1].kernel_size, 
                         stride = block.feature[1].stride, padding = block.feature[1].padding, 
                         groups = block.feature[1].groups, dilation = block.feature[1].dilation, bias = False)
        if block.feature[2].bias != None:
            restore = nn.Conv2d(rankout, block.feature[2].out_channels, kernel_size = 1, bias = True)
            restore.bias.data = block.feature[2].bias.data
        else:
            restore = nn.Conv2d(rankout, block.feature[2].out_channels, kernel_size = 1, bias = False)
        
        c, [t, s] = td.partial_tucker(core_weight, modes = [0, 1], rank = [rankout, rankin])

        s = np.transpose(s)
        # print(compress_weight.shape)
        compress_weight = np.reshape(compress_weight, (compress_weight.shape[0], -1))
        # print(compress_weight.shape)
        # print(s.shape)
        s = np.matmul(s, compress_weight)
        s = np.reshape(s, (s.shape[0], -1, 1, 1))
        compress.weight.data = torch.from_numpy(s.copy())
        
        core.weight.data = torch.from_numpy(c.copy())
        # print("-----")
        restore_weight = np.reshape(restore_weight, (restore_weight.shape[0], -1))
        # print(restore_weight.shape)
        # print(t.shape)
        t = np.matmul(restore_weight, t)
        t = np.reshape(t, (t.shape[0], -1, 1, 1))
        # print(t.shape)
        # print("-----")
        restore.weight.data = torch.from_numpy(t.copy())

        layers = [compress, core, restore]

        self.feature = nn.Sequential(*layers)

    def weakened_rank(self, weight, reduction_rate):
        '''
        extreme_in_rank = vbmf.EVBMF(tensorly.unfold(weight, 1))[1].shape[0]
        extreme_out_rank = vbmf.EVBMF(tensorly.unfold(weight, 0))[1].shape[0]

        init_in_rank, init_out_rank = TuckerBlock.exact_rank(weight)
        
        weakened_in_rank = init_in_rank - int(reduction_rate * (init_in_rank - extreme_in_rank))
        weakened_out_rank = init_out_rank - int(reduction_rate * (init_out_rank - extreme_out_rank))
        '''

        init_in_rank, init_out_rank = TuckerBlock.rank_select(weight, 'exact')

        weakened_in_rank = int(init_in_rank * (1. - reduction_rate))
        weakened_out_rank = int(init_out_rank - (1. - reduction_rate))

        return weakened_in_rank, weakened_out_rank
        
    def forward(self,x):
        return self.feature(x)

# proposed method in "CP-decomposition with Tensor Power Method for Convolutional Neural Networks Compression"
# CPBlock hasn't been refined and tested
class CPBlock(nn.Module):
    ''' conv_weight, conv_bias: numpy '''
    def __init__(self, layer, rank=5):
        super(CPBlock, self).__init__()

        down = nn.Conv2d(layer.in_channels, rank, kernel_size = 1, bias = False)
        spatio = nn.Conv2d(rank, rank, kernel_size = layer.kernel_size, stride = layer.stride, padding = layer.padding, groups = rank, bias = False)
        # print(torch.stack(spatio.weight.chunk(rank, 0)).shape)
        if layer.bias == None:
            restore = nn.Conv2d(rank, layer.out_channels, kernel_size = 1, bias = False)
        else:
            restore = nn.Conv2d(rank, layer.out_channels, kernel_size = 1, bias = True)
            restore.bias.data = layer.bias.data

        weight = layer.weight.data.numpy()
        weight = weight.reshape((weight.shape[0], weight.shape[1], -1))
        # normalize_factors = False, so that scalar return is all one
        print("decomposing")
        scalar, [_out, _in, kernel] = td.parafac(weight, rank , n_iter_max = 100, init = 'random') # , normalize_factors = True)
        # _out, _in, kernel = CPBlock.Tensor_Power_Method(weight, rank)

        _in = np.transpose(_in)
        # _in = np.array([_in[i] * scalar[i] for i in range(rank)])
        _in = np.reshape(_in, (rank, layer.in_channels, 1, 1))
        down.weight.data = torch.from_numpy(_in.copy())

        kernel = np.transpose(kernel)
        kernel = np.reshape(kernel, (rank, 1, layer.kernel_size[0], layer.kernel_size[1]))
        spatio.weight.data = torch.from_numpy(kernel.copy())

        _out = np.reshape(_out, (layer.out_channels, rank, 1, 1))
        restore.weight.data = torch.from_numpy(_out.copy())

        self.feature = nn.Sequential(down, spatio, restore)

    @staticmethod
    def Tensor_Power_Method(weight, rank):
        _out, _in, kernel = [], [], []

        print("itering")
        for i in range(rank):
            scalar, [temp_out, temp_in, temp_kernel] = td.parafac(weight, 1, n_iter_max = 100) # , init = 'random') # , normalize_factors = True
            _out.append(temp_out)
            _in.append(temp_in)
            kernel.append(temp_kernel)
            weight = weight - tensorly.cp_tensor.cp_to_tensor([scalar, [temp_out, temp_in, temp_kernel]])
            print(np.linalg.norm(weight))
        print("iter end")

        return np.array(_out), np.array(_in), np.array(kernel)

    def forward(self, x):
        return self.feature(x)
        