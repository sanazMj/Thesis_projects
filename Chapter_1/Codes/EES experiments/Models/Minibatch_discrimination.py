import torch
import torch.nn as nn
import torch.nn.init as init

class MinibatchDiscrimination(nn.Module):
    def __init__(self, in_features, out_features, kernel_dims,Minibatch_kind = 'L1 Norm', mean=False):
        super(MinibatchDiscrimination, self).__init__()
        self.Minibatch_kind = Minibatch_kind
        self.in_features = in_features
        self.out_features = out_features
        self.kernel_dims = kernel_dims
        self.mean = mean
        self.T = nn.Parameter(torch.Tensor(in_features, out_features, kernel_dims))
        init.normal(self.T, 0, 1)

    def forward(self, x):
        # x is NxA
        # T is AxBxC
        # print(x.shape)
        # print(self.T.view(self.in_features, -1).shape)
        matrices = x.mm(self.T.view(self.in_features, -1))
        # print(matrices.shape)
        matrices = matrices.view(-1, self.out_features, self.kernel_dims)

        M = matrices.unsqueeze(0)  # 1xNxBxC
        M_T = M.permute(1, 0, 2, 3)  # Nx1xBxC


        # print('temp',temp[0])
        if self.Minibatch_kind == 'L1 Norm':
            norm = torch.abs(M - M_T).sum(3)
            expnorm = torch.exp(-norm)
        elif self.Minibatch_kind == 'L2 Norm':
            norm = ((torch.abs(M - M_T))**2).sum(3)
            expnorm = torch.exp(-norm)
        elif self.Minibatch_kind == 'identical':
            norm = torch.abs(M - M_T).sum(3)
            expnorm = torch.exp(-norm*100)
        elif self.Minibatch_kind == 'Cosine':
            cos1 = torch.nn.CosineSimilarity(dim=3)
            norm = cos1(M, M_T)
            expnorm = torch.exp(-norm)


        o_b = (expnorm.sum(0) - 1)   # NxB, subtract self distance
        # o_b = (exp_Sum - 1)
        if self.mean:
            o_b /= x.size(0) - 1
        # print('minibatch ob ', o_b.shape)
        x = torch.cat([x, o_b], 1)
        return x