import torch
from torch import nn, optim
Print_flag = False
class DiscriminatorNet(torch.nn.Module):
    """
    A three hidden-layer discriminative neural network
    """

    def __init__(self, n_features, ndf, minibatch=False, Pack_num=1):
        super(DiscriminatorNet, self).__init__()
        n_out = 1
        self.minibatch_layer = minibatch
        self.Pack_num = Pack_num
        self.hidden0 = nn.Sequential(
            nn.Linear(n_features * self.Pack_num, ndf),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )
        self.hidden1 = nn.Sequential(
            nn.Linear(ndf, ndf//2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )
        self.hidden2 = nn.Sequential(
            nn.Linear(ndf//2, ndf//4),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )
        if self.minibatch_layer:
            self.out = nn.Sequential(
                torch.nn.Linear(ndf//8 + ndf//4, n_out),
                torch.nn.Sigmoid()
            )
        else:
            self.out = nn.Sequential(
                torch.nn.Linear(ndf//4, n_out),
                torch.nn.Sigmoid()
            )
        # self.out = nn.Sequential(
        #     torch.nn.Linear(ndf//4, n_out),
        #     torch.nn.Sigmoid() # For WGAN is commented
        # )

    def forward(self, x, y):
        # x = torch.cat((x, y), 1)
        if Print_flag:
            print('d', x.shape)
        x = self.hidden0(x)
        if Print_flag:
            print('d', x.shape)
        x = self.hidden1(x)
        if Print_flag:
            print('d', x.shape)
        x = self.hidden2(x)
        if Print_flag:
            print('d', x.shape)
        if self.minibatch_layer:
            x = self.minibatch_layer(x)
            if Print_flag:
                print(x.shape)
        x = self.out(x)
        if Print_flag:
            print('d', x.shape)
        return x


class GeneratorNet(torch.nn.Module):
    """
    A three hidden-layer generative neural network
    """

    def __init__(self, num_condition, num_pixels, ngf, zdim):
        super(GeneratorNet, self).__init__()
        n_features = zdim + 0
        n_out = num_pixels

        self.hidden0 = nn.Sequential(
            nn.Linear(n_features, ngf),
            nn.LeakyReLU(0.2)
        )
        self.hidden1 = nn.Sequential(
            nn.Linear(ngf, ngf*2),
            nn.LeakyReLU(0.2)
        )
        self.hidden2 = nn.Sequential(
            nn.Linear(ngf*2, ngf*4),
            nn.LeakyReLU(0.2)
        )

        self.out = nn.Sequential(
            nn.Linear(ngf*4, n_out),
            nn.Sigmoid()
        )

    def forward(self, x, y):
        if Print_flag:
            print('g', x.shape)
        # x = torch.cat((x, y), 1)
        x = self.hidden0(x)
        if Print_flag:
            print('g', x.shape)
        x = self.hidden1(x)
        if Print_flag:
            print('g', x.shape)
        x = self.hidden2(x)
        if Print_flag:
            print('g', x.shape)
        x = self.out(x)
        if Print_flag:
            print('g', x.shape)
        return x
class VARNet(torch.nn.Module):
    """
    A three hidden-layer discriminative neural network
    """

    def __init__(self, n_features, ndf, minibatch=False, Pack_num=1):
        super(VARNet, self).__init__()
        n_out = 1
        self.minibatch_layer = minibatch
        self.Pack_num = Pack_num
        self.hidden0 = nn.Sequential(
            nn.Linear(n_features * self.Pack_num, ndf),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )
        self.hidden1 = nn.Sequential(
            nn.Linear(ndf, ndf//2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )
        self.hidden2 = nn.Sequential(
            nn.Linear(ndf//2, ndf//4),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )
        if self.minibatch_layer:
            self.out = nn.Sequential(
                torch.nn.Linear(ndf//8 + ndf//4, n_out),
                torch.nn.Sigmoid()
            )
        else:
            self.out = nn.Sequential(
                torch.nn.Linear(ndf//4, n_out),
                torch.nn.Sigmoid()
            )
        # self.out = nn.Sequential(
        #     torch.nn.Linear(ndf//4, n_out),
        #     torch.nn.Sigmoid() # For WGAN is commented
        # )

    def forward(self, x, y):
        # x = torch.cat((x, y), 1)
        if Print_flag:
            print('v', x.shape)
        x = self.hidden0(x)
        if Print_flag:
            print('v', x.shape)
        x = self.hidden1(x)
        if Print_flag:
            print('v', x.shape)
        x = self.hidden2(x)
        if Print_flag:
            print('v', x.shape)
        if self.minibatch_layer:
            x = self.minibatch_layer(x)
            if Print_flag:
                print(x.shape)
        x = self.out(x)
        if Print_flag:
            print('v', x.shape)
        return x

def initialize_models(num_features, num_condition, num_pixels, ndf, ngf, zdim, minibatch_net, pack_num):

    discriminator = DiscriminatorNet(num_pixels, ndf, minibatch_net, 1)
    varnet = VARNet(num_pixels, ndf, minibatch_net, pack_num)
    generator = GeneratorNet(num_condition, num_pixels, ngf, zdim)
    if torch.cuda.is_available():
        discriminator.cuda()
        generator.cuda()
        varnet.cuda()

    # Optimizers
    d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002)
    g_optimizer = optim.Adam(generator.parameters(), lr=0.0002)
    varnet_optimizer = optim.Adam(varnet.parameters(), lr=0.0002)

    # Loss function (Creates a criterion that measures the Binary Cross Entropy
    # between the target and the output)
    loss = nn.BCELoss()


    return generator, discriminator, varnet, g_optimizer, d_optimizer, varnet_optimizer, loss


