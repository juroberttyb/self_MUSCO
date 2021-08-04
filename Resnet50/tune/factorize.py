import numpy as np
import torch
import torch.nn as nn
import tensorly.decomposition as td
import tensorly

def gating(gate, threshold =0.01): # 0.01
    left = []
    for i in range(gate.shape[0]):
        if abs(gate[i]) >= threshold:
            left.append(i)
    if len(left) == 0:
        print('exist residual link to be completly removed!!!')
        left = [np.argmax(np.abs(gate))]
    
    # print('gate: ' + str(gate.shape) + ' -> '  + str(len(left)))
    return left

class SVDBlock(nn.Module):
    ''' conv_weight, conv_bias: numpy '''
    def __init__(self, weight, stride = 1, threshold = 0.):
        super(SVDBlock, self).__init__()

        u, s, v = self.ToTorchForm(weight)

        self.element = nn.utils.weight_norm(nn.Conv2d(u.shape[1], u.shape[0], kernel_size = 1, stride = stride, bias = False))
        self.restore = nn.utils.weight_norm(nn.Conv2d(v.shape[1], v.shape[0], kernel_size = 1, bias = False))

        self.element.weight_v.data = torch.from_numpy(u.copy())
        self.element.weight_g.data = torch.from_numpy(s.copy())

        self.restore.weight_v.data = torch.from_numpy(v.copy())
        self.restore.weight_g.data = torch.from_numpy(np.array([np.linalg.norm(v[i]) for i in range(v.shape[0])])) # .copy()

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
        x = self.element(x)
        x = self.restore(x)
            
        return x

class TuckerBlock(nn.Module):
    ''' conv_weight, conv_bias: numpy '''
    def __init__(self, weight, stride = 1, padding = 1, groups = 1, dilation = 1):
        super(TuckerBlock, self).__init__()

        # chin, chout = self.complete_rank(weight)
        chin, chout = self.exact_rank(weight) # , tol = 0.333)
        
        self.compress = nn.utils.weight_norm(nn.Conv2d(chin, chin, kernel_size = 1, bias = False))
        self.core = nn.utils.weight_norm(nn.Conv2d(chin, chout, kernel_size = weight.shape[2], stride = stride, groups = groups, padding = padding, dilation = dilation, bias = False))
        # self.restore = nn.Conv2d(chout, chout, kernel_size = 1, bias = False)
        self.restore = nn.utils.weight_norm(nn.Conv2d(chout, chout, kernel_size = 1, bias = False))

        c, [t, s] = td.partial_tucker(weight, modes = [0, 1], ranks = [chout, chin])

        s = np.transpose(s)
        s = np.reshape(s, (s.shape[0], -1, 1, 1))
        self.compress.weight_v.data = torch.from_numpy(s.copy())
        self.compress.weight_g.data = torch.from_numpy(np.array([np.linalg.norm(s[i]) for i in range(s.shape[0])]))
        
        self.core.weight_v.data = torch.from_numpy(c.copy())
        self.core.weight_g.data = torch.from_numpy(np.array([np.linalg.norm(c[i]) for i in range(c.shape[0])]))

        t = np.reshape(t, (t.shape[0], -1, 1, 1))
        # self.restore.weight.data = torch.from_numpy(t.copy())
        self.restore.weight_v.data = torch.from_numpy(t.copy())
        self.restore.weight_g.data = torch.from_numpy(np.array([np.linalg.norm(t[i]) for i in range(t.shape[0])]))

    def exact_rank(self, tensor, tol = None):
        in_rank = np.linalg.matrix_rank(tensorly.unfold(tensor, 1), tol = tol)
        out_rank = np.linalg.matrix_rank(tensorly.unfold(tensor, 0), tol = tol)
        
        return in_rank, out_rank

    def complete_rank(self, weight):
        return weight.shape[1], weight.shape[0]

    def forward(self,x):
        x = self.compress(x)
        x = self.core(x)
        x = self.restore(x)
            
        return x

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

class OutputSVD(nn.Module):
    ''' conv_weight, conv_bias: numpy '''
    def __init__(self, net, stride = 1):
        super(OutputSVD, self).__init__()

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
            '''
            conv = nn.Conv2d(input_dim, output_dim, kernel_size = 1, stride = stride, bias = False)

            u, gate = net.element.weight_v.data.numpy(), net.element.weight_g.data.numpy()
            u = np.array([u[i] / np.linalg.norm(u[i]) * gate[i] for i in range(u.shape[0])])

            v, gate = net.restore.weight_v.data.numpy(), net.restore.weight_g.data.numpy()
            v = np.array([v[i] / np.linalg.norm(v[i]) * gate[i] for i in range(v.shape[0])])

            # v = net.restore.weight.data.numpy()

            weight = self.recover(u, v)

            conv.weight.data = torch.from_numpy(weight.copy())

            layers = [conv]
            '''
            layers = [net.element, net.restore]

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

class OutputTucker(nn.Module):
    ''' conv_weight, conv_bias: numpy '''
    def __init__(self, net, stride = 1, padding = 1, dilation = 1):
        super(OutputTucker, self).__init__()

        cin, cout = gating(net.compress.weight_g.data.numpy()), gating(net.core.weight_g.data.numpy())
        rankin, rankout, kernel_size = len(cin), len(cout), net.core.weight_v.data.numpy().shape[2]
        input_dim, output_dim = net.compress.weight_v.data.numpy().shape[1], net.restore.weight.data.numpy().shape[0]
        
        if self.faster_check(rankin, rankout, kernel_size, input_dim, output_dim):
            compress = nn.Conv2d(input_dim, rankin, kernel_size = 1, bias = False)
            core = nn.Conv2d(rankin, rankout, kernel_size = kernel_size, stride = stride, padding = padding, dilation = dilation, bias = False)
            restore = nn.Conv2d(rankout, output_dim, kernel_size = 1, bias = False)

            weight, gate = net.compress.weight_v.data.numpy(), net.compress.weight_g.data.numpy()
            weight, gate = weight[cin, :, :, :], gate[cin]
            weight = np.array([weight[i] / np.linalg.norm(weight[i]) * gate[i] for i in range(weight.shape[0])])

            compress.weight.data = torch.from_numpy(weight.copy())

            weight, gate = net.core.weight_v.data.numpy(), net.core.weight_g.data.numpy()
            weight = np.array([weight[i] / np.linalg.norm(weight[i]) * gate[i] for i in range(weight.shape[0])])
            weight = weight[:, cin, :, :]
            weight = weight[cout, :, :, :]

            core.weight.data = torch.from_numpy(weight.copy())

            weight, gate = net.restore.weight_v.data.numpy(), net.restore.weight_g.data.numpy()
            weight = np.array([weight[i] / np.linalg.norm(weight[i]) * gate[i] for i in range(weight.shape[0])])
            weight = weight[:, cout, :, :]

            restore.weight.data = torch.from_numpy(weight.copy())

            layers = [compress, core, restore]
        else:
            '''
            conv = nn.Conv2d(input_dim, output_dim, kernel_size = kernel_size, stride = stride, padding = padding, dilation = dilation, bias = False)

            s, gate = net.compress.weight_v.data.numpy(), net.compress.weight_g.data.numpy()
            s = np.array([s[i] / np.linalg.norm(s[i]) * gate[i] for i in range(s.shape[0])])

            c, gate = net.core.weight_v.data.numpy(), net.core.weight_g.data.numpy()
            c = np.array([c[i] / np.linalg.norm(c[i]) * gate[i] for i in range(c.shape[0])])

            t, gate = net.restore.weight_v.data.numpy(), net.restore.weight_g.data.numpy()
            t = np.array([t[i] / np.linalg.norm(t[i]) * gate[i] for i in range(t.shape[0])])

            weight = self.recover(c, t, s, kernel_size)

            conv.weight.data = torch.from_numpy(weight.copy())

            layers = [conv]
            '''
            layers = [net.compress, net.core, net.restore]

        self.feature = nn.Sequential(*layers)

    def faster_check(self, rankin, rankout, kernel_size, input_dim, output_dim):
        if rankin * input_dim + rankin * kernel_size * kernel_size * rankout + rankout * output_dim < input_dim * kernel_size * kernel_size * output_dim:
            return True
        return False

    def recover(self, c, t, s, kernel_size):
        s = np.reshape(s, (s.shape[0], -1))
        s = np.transpose(s)

        t = np.reshape(t, (t.shape[0], -1))

        weight = tensorly.tucker_tensor.tucker_to_tensor((c, [t, s, np.eye(kernel_size), np.eye(kernel_size)]))
        # print(s.dtype)
        # print(weight.dtype)
        weight = np.array(weight, dtype = np.float32)

        return weight

    def forward(self,x):
        x = self.feature(x)
        
        return x
