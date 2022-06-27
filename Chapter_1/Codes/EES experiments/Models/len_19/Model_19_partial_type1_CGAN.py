
'Works for partial images on 19*19 EES dataset'

import torch
from torch import nn, optim
import numpy as np
from torch.autograd.variable import Variable
# from Loss import LossBGANConGen, LossBGANConDis
print_flag = False
from Minibatch_discrimination import *

class DiscriminatorNet(torch.nn.Module):

    def __init__(self,channels,factor, Kernel_factor,z_dim,num_condition,num_pixels, Minibatch=False, Pack_num=1):
        super(DiscriminatorNet, self).__init__()
        self.minibatch = Minibatch
        self.z_dim = z_dim
        self.num_condition = 0
        self.num_pixels = num_pixels
        self.pixels_edge = int(np.sqrt(num_pixels))
        self.channels = channels//factor
        self.Pack_num = Pack_num
        self.init_pixel = 2
        self.final_pixel = self.pixels_edge
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1 + self.num_condition, out_channels=self.channels//8, kernel_size=3,
                stride=1, padding=0, bias=False
            ),
            nn.BatchNorm2d(self.channels//8),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=self.channels//8, out_channels=self.channels//4, kernel_size=3,
                stride=1, padding=0, bias=False
            ),
            nn.BatchNorm2d(self.channels//4),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=self.channels//4, out_channels=self.channels//2, kernel_size=3,
                stride=1, padding=0, bias=False
            ),
            nn.BatchNorm2d(self.channels//2),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(
                in_channels=self.channels // 2, out_channels=self.channels, kernel_size=3,
                stride=1, padding=0, bias=False
            ),
            nn.BatchNorm2d(self.channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        if Minibatch:
            self.linear1 = nn.Sequential(
                nn.Linear(channels * 2 * self.init_pixel * self.init_pixel, 1),
                nn.Sigmoid()
            )
        else:
            self.linear1 = nn.Sequential(
                nn.Linear(self.channels * self.init_pixel * self.init_pixel, 1),
                nn.Sigmoid()
            )

    def forward(self, x, y):
        # Convolutional layers
        if print_flag:
            print('d',x.shape)
        x = x.reshape(int(x.shape[0]/self.Pack_num ),1*self.Pack_num ,self.pixels_edge,self.pixels_edge)
        if print_flag:
            print('d',x.shape)

        # y1 = Variable(torch.randn(y.shape[0],  self.pixels_edge, self.pixels_edge,y.shape[1] ))
        # for i in range(y1.shape[0]):
            # for j in range(y1.shape[1]):
            #     for k in range(y1.shape[2]):
            #         y1[i] = torch.tensor(y[i])
        # y1= torch.tensor(np.stack([np.tile(y[i].cpu(),(self.final_pixel,self.final_pixel,1)) for i in range(y.shape[0])]))
        # # y1 = torch.tensor(np.stack([y[i] for i in range(y.shape[0])]))
        # if torch.cuda.is_available(): y1 = y1.cuda()

        # y = y.reshape(y.shape[0], y.shape[3],y.shape[1], y.shape[2])
        # if print_flag:
        #     print('d',x.shape,y.shape)
        # #
        # x = torch.cat([x, y], 1)

        x = self.conv1(x)
        if print_flag:
            print('d', x.shape)

        x = self.conv2(x)
        if print_flag:
            print('d', x.shape)

        x = self.conv3(x)
        if print_flag:
            print('d', x.shape)

        x = self.conv4(x)
        if print_flag:
            print('d', x.shape)
        x = x.reshape(-1,x.shape[1]*x.shape[2]*x.shape[3])
        if print_flag:
            print('d', x.shape)        #
        if self.minibatch:
            x = self.minibatch(x)
        if print_flag:
            print('d', x.shape)

        x = self.linear1(x)
        if print_flag:
            print('d', x.shape)
        return x




class GeneratorNet(torch.nn.Module):

    def __init__(self, channels,factor, Kernel_factor, z_dim, num_condition, num_pixels):
        self.z_dim = z_dim
        self.num_pixels = num_pixels
        self.pixels_edge = int(np.sqrt(num_pixels))
        self.num_condition = 0
        self.channels = channels//factor
        self.init_pixel = 2
        # self.init_pixel = self.pixels_edge

        super(GeneratorNet, self).__init__()

        self.linear1 = nn.Sequential(
            nn.Linear(self.z_dim + self.num_condition, self.init_pixel * self.init_pixel* self.channels),
            nn.BatchNorm1d(self.init_pixel * self.init_pixel* self.channels),
            nn.LeakyReLU(negative_slope=0.2,inplace=True)
        )

        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(self.channels, self.channels//2,kernel_size=3,
                stride=1, padding=0, bias = False),
            nn.BatchNorm2d(self.channels//2),
            nn.LeakyReLU(negative_slope=0.2,inplace=True)
        )
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(self.channels//2, self.channels // 4, kernel_size=3,
                               stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.channels//4),
            nn.LeakyReLU(negative_slope=0.2,inplace=True)
        )
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(self.channels // 4, self.channels // 8, kernel_size=3,
                               stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.channels // 8),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        self.deconv4 = nn.Sequential(
            nn.ConvTranspose2d(self.channels//8, 1, kernel_size=3,
                               stride=1, padding=0, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )


    def forward(self, x, y):
        if print_flag:
            print('g',x.shape,)

        x = x.reshape(x.size(0), x.size(1))
        if print_flag:
            print('g', x.shape)

        # x = torch.cat([x, y], 1)
        # if print_flag:
        #     print('g', x.shape)

        #Linear Layer
        x = self.linear1(x)
        if print_flag:
            print('g', x.shape)

        x = x.reshape(-1,self.channels,self.init_pixel, self.init_pixel)
        if print_flag:
            print('g', x.shape)
        # Convolutional layers
        x = self.deconv1(x)
        if print_flag:
            print('g', x.shape)
        x = self.deconv2(x)
        if print_flag:
            print('g', x.shape)        #
        x = self.deconv3(x)
        if print_flag:
            print('g', x.shape)
        x = self.deconv4(x)
        if print_flag:
            print('g', x.shape)
        return x


#
def initialize_models(channels,factor, Kernel_factor,z_dim, num_condition, num_pixels,Minibatch,minibatch_kind, Pack_num):
    init_pixel = 2
    if Minibatch:
        minibatch_net = MinibatchDiscrimination(channels * init_pixel * init_pixel,
                                                channels * init_pixel * init_pixel, 5,Minibatch_kind=minibatch_kind)
        print('Model with minibatch')
    else:
        minibatch_net = False

    discriminator = DiscriminatorNet(channels,factor, Kernel_factor,z_dim, num_condition,num_pixels,minibatch_net, Pack_num)
    generator = GeneratorNet(channels,factor, Kernel_factor,z_dim, num_condition, num_pixels)

    if torch.cuda.is_available():
        discriminator.cuda()
        generator.cuda()

    # Optimizers
    d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002)
    g_optimizer = optim.Adam(generator.parameters(), lr=0.0002)

    # Loss function (Creates a criterion that measures the Binary Cross Entropy
    # between the target and the output)
    loss_BCE = nn.BCELoss()
    # loss_g = LossBGANConGen()
    # loss_d = LossBGANConDis()
    return generator, discriminator, g_optimizer, d_optimizer, loss_BCE

# def initialize_models(z_dim,optim_betas, unrolled_steps, d_learning_rate, g_learning_rate, num_features, num_condition, num_pixels):
#     discriminator = DiscriminatorNet(num_condition,num_pixels)
#     generator = GeneratorNet(z_dim, num_condition, num_pixels)
#     if torch.cuda.is_available():
#         discriminator.cuda()
#         generator.cuda()
#
#     # Optimizers
#     d_optimizer = optim.Adam(discriminator.parameters(), lr=d_learning_rate,betas=optim_betas)
#     g_optimizer = optim.Adam(generator.parameters(), lr=g_learning_rate,betas=optim_betas)
#
#     # Loss function (Creates a criterion that measures the Binary Cross Entropy
#     # between the target and the output)
#     loss = nn.BCELoss()
#
#     return generator, discriminator, g_optimizer, d_optimizer, loss