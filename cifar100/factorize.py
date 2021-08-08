import numpy as np
import torch
import torch.nn as nn
import tensorly.decomposition as td
import tensorly
import vbmf

# 待修改MuscoSVD, CPBlock

class SVDBlock(nn.Module):
    ''' conv_weight, conv_bias: numpy '''
    def __init__(self, net, stride = 1, threshold = 0., bias = False):
        super(SVDBlock, self).__init__()
        
        weight = net.weight.data.numpy()

        u, v = self.TorchFormSVD(weight, threshold)

        element = nn.Conv2d(u.shape[1], u.shape[0], kernel_size = 1, stride = stride, bias = False)
        restore = nn.Conv2d(v.shape[1], v.shape[0], kernel_size = 1, bias = bias)

        element.weight.data = torch.from_numpy(u.copy())
        restore.weight.data = torch.from_numpy(v.copy())

        if bias:
            restore.bias.data = net.bias.data

        self.feature = nn.Sequential(element, restore)

    # filters: weights of 1x1 conv layer
    def TorchFormSVD(self, filters, threshold, with_s=False):
        filters = np.transpose(filters.reshape((filters.shape[0], -1)))

        if with_s:
            u, s, v = self.SVD(filters, with_s=with_s, threshold=threshold)
        else:
            u, v = self.SVD(filters, with_s=with_s, threshold=threshold)
        
        u = u.transpose()
        u = u.reshape((*u.shape, 1, 1))
        
        v = v.transpose()
        v = v.reshape((*v.shape, 1, 1))

        # s = np.sqrt(s)
        if with_s:
            s = s.reshape(s.shape[0], 1, 1, 1)
            return u, s, v
        return u, v

    # weight: numpy array of shape chin * chout
    def SVD(self, weight, with_s, threshold):
        u, s, v = np.linalg.svd(weight, full_matrices=False)
        # full_matrices=False -> the shapes are (..., M, K) and (..., K, N), respectively, where K = min(M, N)

        for i in range(s.shape[0]):
            if s[i] < threshold:
                s = s[:i]
                break
        
        u = u[:, :s.shape[0]]
        
        v = v[:s.shape[0], :]
        
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

class MuscoSVD(nn.Module):
    ''' conv_weight, conv_bias: numpy '''
    def __init__(self, net, reduction_rate, stride=1, bias=False):
        super(MuscoSVD, self).__init__()

        element_weight = net.feature[0].weight.data.numpy()
        restore_weight = net.feature[1].weight.data.numpy()

        rank = ...

        filters = self.TorchReform(element_weight, restore_weight)

    def TorchReform(self, u, v):

        u = u.reshape((u.shape[0], -1))
        u = u.transpose()
        
        v = v.reshape((v.shape[0], -1))
        v = v.transpose()

        filters = np.matmul(u, v)
        filters = filters.transpose()
        filters = filters.reshape((*filters.shape, 1, 1))

        return filters

    def forward(self, x):
        return self.feature(x)

class TuckerBlock(nn.Module):
    ''' conv_weight, conv_bias: numpy '''
    def __init__(self, net, padding = 'same', bias = False):
        super(TuckerBlock, self).__init__()

        weight = net.weight.data.numpy()

        rank_in, rank_out = self.complete_rank(weight)
        # rank_in, rank_out = self.weakened_rank(weight)

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
            restore.bias.data = net.bias.data

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

class MuscoTucker(nn.Module):
    ''' conv_weight, conv_bias: numpy '''
    def __init__(self, net, reduction_rate, padding = 'same', bias = False):
        super(MuscoTucker, self).__init__()

        compress_weight = net.feature[0].weight.data.numpy()
        core_weight = net.feature[1].weight.data.numpy()
        restore_weight = net.feature[2].weight.data.numpy()
        
        rankin, rankout = self.weakened_rank(core_weight, reduction_rate)
        kernel_size = core_weight.shape[2]
        input_dim, output_dim = compress_weight.shape[1], restore_weight.shape[0]

        compress = nn.Conv2d(input_dim, rankin, kernel_size = 1, bias = False)
        core = nn.Conv2d(rankin, rankout, kernel_size = kernel_size, padding = padding, bias = False)
        restore = nn.Conv2d(rankout, output_dim, kernel_size = 1, bias = bias)
        
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

        if bias:
            restore.bias.data = net.feature[2].bias.data

        layers = [compress, core, restore]

        self.feature = nn.Sequential(*layers)

    def weakened_rank(self, weight, reduction_rate):
        extreme_in_rank = vbmf.EVBMF(tensorly.unfold(weight, 1))[1].shape[0]
        extreme_out_rank = vbmf.EVBMF(tensorly.unfold(weight, 0))[1].shape[0]
        
        init_in_rank = weight.shape[1]
        init_out_rank = weight.shape[0]
        
        weakened_in_rank = init_in_rank - int(reduction_rate * (init_in_rank - extreme_in_rank))
        weakened_out_rank = init_out_rank - int(reduction_rate * (init_out_rank - extreme_out_rank))
        
        return weakened_in_rank, weakened_out_rank
        
    def forward(self,x):
        return self.feature(x)

class CPBlock(nn.Module):
    ''' conv_weight, conv_bias: numpy '''
    def __init__(self, weight, stride = 1, padding = 1, groups = 1, dilation = 1):
        super(CPBlock, self).__init__()

        kernel_size = weight.shape[2]
        # weight = np.reshape(weight, (weight.shape[0], weight.shape[1], kernel_size * kernel_size))

        cin, cout = weight.shape[1], weight.shape[0]
        r = self.approx_rank(weight)
        
        self.compress = nn.Conv2d(cin, r, kernel_size = 1, stride = stride, bias = False)
        self.right = nn.Conv2d(r, r, kernel_size = (kernel_size, 1), padding = (padding, 0), groups = r, bias = False)
        self.down = nn.Conv2d(r, r, kernel_size = (1, kernel_size), padding = (0, padding), groups = r, bias = False)
        self.restore = nn.Conv2d(r, cout, kernel_size = 1, bias = False)

        # normalize_factors = False, so that scalar return is all one
        scalar, [_out, _in, kernel1, kernel2] = td.parafac(weight, r, n_iter_max = 100, normalize_factors = True)

        _in = np.transpose(_in)
        _in = np.array([_in[i] * scalar[i] for i in range(r)])
        _in = np.reshape(_in, (r, -1, 1, 1))
        self.compress.weight.data = torch.from_numpy(_in.copy())

        kernel1 = np.transpose(kernel1)
        kernel1 = np.reshape(kernel1, (-1, 1, kernel_size, 1))
        self.right.weight.data = torch.from_numpy(kernel1.copy())

        kernel2 = np.transpose(kernel2)
        kernel2 = np.reshape(kernel2, (-1, 1, 1, kernel_size))
        self.down.weight.data = torch.from_numpy(kernel2.copy())

        _out = np.reshape(_out, (-1, r, 1, 1))
        self.restore.weight.data = torch.from_numpy(_out.copy())

    def approx_rank(self, weight):
        return 48 # weight.shape[0] * weight.shape[2] # * weight.shape[1] # * weight.shape[3]
        
    def forward(self,x):
        x = self.compress(x)
        x = self.right(x)
        x = self.down(x)
        x = self.restore(x)
            
        return x

def TuckerFactorze(net):
    for e in dir(net):
        layer = getattr(net, e)
        if isinstance(layer, nn.Conv2d):
            # print("get conv2d layer " + e)

            shape = layer.weight.data.numpy().shape
            if (shape[1] > 3 and shape[2] > 1 and shape[3] > 1):
                # print(e + " is a worthy layer")
                
                setattr(net, e, TuckerBlock(layer, bias = True))
    return net

def TuckerMuscoStep(net, reduction_rate = 0.2):
    for e in dir(net):
        layer = getattr(net, e)
        if isinstance(layer, TuckerBlock) or isinstance(layer, MuscoTucker):
            # print("get Block " + e)
            
            setattr(net, e, MuscoTucker(layer, reduction_rate = reduction_rate, bias = True))
    return net
