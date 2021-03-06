import numpy as np
import torch
import torch.nn as nn
import tensorly.decomposition as td
import tensorly
import vbmf

w1 = 0.5 # model/model2 0.7
w2 = 0.21



class TuckerBlock(nn.Module):
    ''' conv_weight, conv_bias: numpy '''
    def __init__(self, weight, stride = 1, padding = 1, groups = 1, dilation = 1):
        super(TuckerBlock, self).__init__()

        # chin, chout = self.complete_rank(weight)
        chin, chout = self.weakened_rank(weight)

        self.compress = nn.Conv2d(chin, chin, kernel_size = 1, bias = False)
        self.core = nn.Conv2d(chin, chout, kernel_size = weight.shape[2], stride = stride, groups = groups, padding = padding, dilation = dilation, bias = False)
        self.restore = nn.Conv2d(chout, chout, kernel_size = 1, bias = False)

        c, [t, s] = td.partial_tucker(weight, modes = [0, 1], rank = [chout, chin])

        s = np.transpose(s)
        s = np.reshape(s, (s.shape[0], -1, 1, 1))
        self.compress.weight.data = torch.from_numpy(s.copy())
        
        self.core.weight.data = torch.from_numpy(c.copy())

        t = np.reshape(t, (t.shape[0], -1, 1, 1))
        self.restore.weight.data = torch.from_numpy(t.copy())

    def exact_rank(self, tensor, tol = None):
        in_rank = int(0.5 * np.linalg.matrix_rank(tensorly.unfold(tensor, 1), tol = tol))
        out_rank = int(0.5 * np.linalg.matrix_rank(tensorly.unfold(tensor, 0), tol = tol))
        # in_rank = vbmf.EVBMF(tensorly.unfold(tensor, 1))[1].shape[0]
        # out_rank = vbmf.EVBMF(tensorly.unfold(tensor, 0))[1].shape[0]
        
        return in_rank, out_rank

    def complete_rank(self, weight):
        return weight.shape[1], weight.shape[0]
    
    def weakened_rank(self, weight):
        extreme_in_rank = vbmf.EVBMF(tensorly.unfold(weight, 1))[1].shape[0]
        extreme_out_rank = vbmf.EVBMF(tensorly.unfold(weight, 0))[1].shape[0]
        
        init_in_rank = weight.shape[1]
        init_out_rank = weight.shape[0]
        
        weakened_in_rank = init_in_rank - int(w1 * (init_in_rank - extreme_in_rank))
        weakened_out_rank = init_out_rank - int(w1 * (init_out_rank - extreme_out_rank))
        
        return weakened_in_rank, weakened_out_rank

    def forward(self,x):
        x = self.compress(x)
        x = self.core(x)
        x = self.restore(x)
            
        return x



class OutputTucker(nn.Module):
    ''' conv_weight, conv_bias: numpy '''
    def __init__(self, net, stride = 1, padding = 1, dilation = 1):
        super(OutputTucker, self).__init__()

        compress_weight = net.compress.weight.data.numpy()
        core_weight = net.core.weight.data.numpy()
        restore_weight = net.restore.weight.data.numpy()
        
        rankin, rankout = self.weakened_rank(core_weight)
        kernel_size = core_weight.shape[2]
        input_dim, output_dim = compress_weight.shape[1], restore_weight.shape[0]

        compress = nn.Conv2d(input_dim, rankin, kernel_size = 1, bias = False)
        core = nn.Conv2d(rankin, rankout, kernel_size = kernel_size, stride = stride, padding = padding, dilation = dilation, bias = False)
        restore = nn.Conv2d(rankout, output_dim, kernel_size = 1, bias = False)
        
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

        layers = [compress, core, restore]

        self.feature = nn.Sequential(*layers)

    def weakened_rank(self, weight):
        extreme_in_rank = vbmf.EVBMF(tensorly.unfold(weight, 1))[1].shape[0]
        extreme_out_rank = vbmf.EVBMF(tensorly.unfold(weight, 0))[1].shape[0]
        
        init_in_rank = weight.shape[1]
        init_out_rank = weight.shape[0]
        
        weakened_in_rank = init_in_rank - int(w2 * (init_in_rank - extreme_in_rank))
        weakened_out_rank = init_out_rank - int(w2 * (init_out_rank - extreme_out_rank))
        
        return weakened_in_rank, weakened_out_rank
        
    def forward(self,x):
        x = self.feature(x)
        
        return x
