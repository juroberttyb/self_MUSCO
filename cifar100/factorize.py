import numpy as np
import torch
import torch.nn as nn
import tensorly.decomposition as td
import tensorly
import vbmf

class SVDBlock(nn.Module):
    ''' conv_weight, conv_bias: numpy '''
    def __init__(self, weight, stride = 1, threshold = 0.):
        super(SVDBlock, self).__init__()

        u, s, v = self.ToTorchForm(weight)

        element = nn.utils.weight_norm(nn.Conv2d(u.shape[1], u.shape[0], kernel_size = 1, stride = stride, bias = False))
        restore = nn.utils.weight_norm(nn.Conv2d(v.shape[1], v.shape[0], kernel_size = 1, bias = False))

        element.weight_v.data = torch.from_numpy(u.copy())
        element.weight_g.data = torch.from_numpy(s.copy())

        restore.weight_v.data = torch.from_numpy(v.copy())
        restore.weight_g.data = torch.from_numpy(np.array([np.linalg.norm(v[i]) for i in range(v.shape[0])])) # .copy()

        self.feature = nn.Sequential(element, restore)

    # filters: weights of 1x1 conv layer
    def ToTorchForm(self, filters):
        filters = np.transpose(filters.reshape((filters.shape[0], -1)))

        u, s, v = self.SVD(filters)
        
        u = u.transpose()
        u = u.reshape((*u.shape, 1, 1))
        
        v = v.transpose()
        v = v.reshape((*v.shape, 1, 1))

        # s = np.sqrt(s)
        s = s.reshape(s.shape[0], 1, 1, 1)

        return u, s, v

    # weight: numpy array of shape chin * chout
    def SVD(self, weight, threshold = 0.):
        u, s, v = np.linalg.svd(weight)

        for i in range(s.shape[0]):
            if s[i] < threshold:
                s = s[:i]
                break
        
        u = u[:, :s.shape[0]]
        
        v = v[:s.shape[0], :]

        '''
        s = np.sqrt(s)

        for i in range(s.shape[0]):
            u[:, i] = u[:, i] * s[i]

        for i in range(s.shape[0]):
            v[i, :] = v[i, :] * s[i]
        '''
        
        return u, s, v

    def forward(self, x):    
        return self.feature(x)

class MuscoSVD(nn.Module):
    ''' conv_weight, conv_bias: numpy '''
    def __init__(self, net, stride = 1):
        super(MuscoSVD, self).__init__()

        cin = gating(net.element.weight_g.data.numpy())
        input_dim, rank, output_dim = net.element.weight_v.data.numpy().shape[1], len(cin), net.restore.weight.data.numpy().shape[0]

        if self.faster_check(input_dim, rank, output_dim):
            element = nn.Conv2d(input_dim, rank, kernel_size = 1, stride = stride, bias = False)
            restore = nn.Conv2d(rank, output_dim, kernel_size = 1, bias = False)

            weight, gate = net.element.weight_v.data.numpy(), net.element.weight_g.data.numpy()
            weight, gate = weight[cin, :, :, :], gate[cin]
            weight = np.array([weight[i] / np.linalg.norm(weight[i]) * gate[i] for i in range(weight.shape[0])])

            element.weight.data = torch.from_numpy(weight.copy())

            weight, gate = net.restore.weight_v.data.numpy(), net.restore.weight_g.data.numpy()
            weight = np.array([weight[i] / np.linalg.norm(weight[i]) * gate[i] for i in range(weight.shape[0])])
            weight = weight[:, cin, :, :]
            restore.weight.data = torch.from_numpy(weight.copy())

            layers = [element, restore]
        else:
            conv = nn.Conv2d(input_dim, output_dim, kernel_size = 1, stride = stride, bias = False)

            u, gate = net.element.weight_v.data.numpy(), net.element.weight_g.data.numpy()
            u = np.array([u[i] / np.linalg.norm(u[i]) * gate[i] for i in range(u.shape[0])])

            v, gate = net.restore.weight_v.data.numpy(), net.restore.weight_g.data.numpy()
            v = np.array([v[i] / np.linalg.norm(v[i]) * gate[i] for i in range(v.shape[0])])

            # v = net.restore.weight.data.numpy()

            weight = self.recover(u, v)

            conv.weight.data = torch.from_numpy(weight.copy())

            layers = [conv]

        self.feature = nn.Sequential(*layers)

    def faster_check(self, m, k, n):
        if m * k + k * n < m * n:
            return True
        return False

    def recover(self, u, v):
        u, v = u.reshape((u.shape[0], -1)).transpose(), v.reshape((v.shape[0], -1)).transpose()
        weight = np.matmul(u, v)
        weight = weight.transpose().reshape((weight.shape[1], -1, 1, 1))

        return weight

    def forward(self, x):
        x = self.feature(x)
            
        return x

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

        block = net.feature

        compress_weight = block[0].weight.data.numpy()
        core_weight = block[1].weight.data.numpy()
        restore_weight = block[2].weight.data.numpy()
        
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

def factorze(net):
    for e in dir(net):
        layer = getattr(net, e)
        if isinstance(layer, nn.Conv2d):
            # print("get conv2d layer " + e)

            shape = layer.weight.data.numpy().shape
            if (shape[1] > 3 and shape[2] > 1 and shape[3] > 1):
                # print(e + " is a worthy layer")
                
                setattr(net, e, TuckerBlock(layer, bias = True))
    return net

def MuscoStep(net, reduction_rate = 0.2):
    for e in dir(net):
        layer = getattr(net, e)
        if isinstance(layer, TuckerBlock) or isinstance(layer, MuscoTucker):
            # print("get Block " + e)
            
            setattr(net, e, MuscoTucker(layer, reduction_rate = reduction_rate, bias = True))
    return net
