import torch
from torch import nn, optim
Print_flag = False
class DiscriminatorNet(torch.nn.Module):
    """
    A three hidden-layer discriminative neural network
    """

    def __init__(self, n_features, ndf, minibatch_net):
        super(DiscriminatorNet, self).__init__()
        n_out = 1
        self.minibatch_layer =minibatch_net
        self.hidden0 = nn.Sequential(
            nn.Linear(n_features, ndf),
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
        x = torch.cat((x, y), 1)
        x = self.hidden0(x)
        x = self.hidden1(x)
        x = self.hidden2(x)
        if self.minibatch_layer:
            x = self.minibatch_layer(x)
            if Print_flag:
                print(x.shape)
        x = self.out(x)
        return x


class GeneratorNet(torch.nn.Module):
    """
    A three hidden-layer generative neural network
    """

    def __init__(self, num_condition, num_pixels, ngf, zdim=100):
        super(GeneratorNet, self).__init__()
        n_features = zdim + num_condition
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
        x = torch.cat((x, y), 1)
        x = self.hidden0(x)
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.out(x)
        return x

def initialize_models(num_features, num_condition, num_pixels, ndf, ngf, zdim, minibatch_net):
    discriminator = DiscriminatorNet(num_features, ndf, minibatch_net)
    generator = GeneratorNet(num_condition, num_pixels, ngf, zdim)
    if torch.cuda.is_available():
        discriminator.cuda()
        generator.cuda()

    # Optimizers
    d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002)
    g_optimizer = optim.Adam(generator.parameters(), lr=0.0002)

    # Loss function (Creates a criterion that measures the Binary Cross Entropy
    # between the target and the output)
    loss = nn.BCELoss()


    return generator, discriminator, g_optimizer, d_optimizer, loss


