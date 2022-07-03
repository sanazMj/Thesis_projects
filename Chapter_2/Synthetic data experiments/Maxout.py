import torch as T


class Maxout(T.nn.Module):
    """Class Maxout implements maxout unit introduced in paper by Goodfellow et al, 2013.

    :param in_feature: Size of each input sample.
    :param out_feature: Size of each output sample.
    :param n_channels: The number of linear pieces used to make each maxout unit.
    :param bias: If set to False, the layer will not learn an additive bias.
    """

    def __init__(self, in_features, out_features, n_channels, bias=True):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.n_channels = n_channels
        self.weight = T.nn.Parameter(T.Tensor(n_channels * out_features, in_features))

        if bias:
            self.bias = T.nn.Parameter(T.Tensor(n_channels * out_features))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def forward(self, input):
        a = T.nn.functional.linear(input, self.weight, self.bias)
        b = T.nn.functional.max_pool1d(a.unsqueeze(-3), kernel_size=self.n_channels)
        return b.squeeze()

    def reset_parameters(self):
        irange = 0.005
        T.nn.init.uniform_(self.weight, -irange, irange)
        if self.bias is not None:
            T.nn.init.uniform_(self.bias, -irange, irange)

    def extra_repr(self):
        return (f'in_features={self.in_features}, '
                f'out_features={self.out_features}, '
                f'n_channels={self.n_channels}, '
                f'bias={self.bias is not None}')