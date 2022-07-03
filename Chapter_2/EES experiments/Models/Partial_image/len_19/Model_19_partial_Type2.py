
'Works for partial images on 19*19 EES dataset'

import torch
from torch import nn, optim
import numpy as np
from torch.autograd.variable import Variable
# from Loss import LossBGANConGen, LossBGANConDis
print_flag = False
class DiscriminatorNet(torch.nn.Module):

    def __init__(self,channels,factor, Kernel_factor,z_dim,num_condition,num_pixels):
        super(DiscriminatorNet, self).__init__()

        self.z_dim = z_dim
        self.num_condition = num_condition
        self.num_pixels = num_pixels
        self.pixels_edge = int(np.sqrt(num_pixels))
        self.channels = channels//factor
        self.init_pixel = 2
        self.final_pixel = self.pixels_edge
        self.conv13 = nn.Sequential(
            nn.Conv2d(
                in_channels=1 + self.num_condition, out_channels=self.channels//8, kernel_size=3,
                stride=1, padding=0, bias=False
            ),
            nn.BatchNorm2d(self.channels//8),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        self.conv15 = nn.Sequential(
            nn.Conv2d(
                in_channels=1 + self.num_condition, out_channels=self.channels // 8, kernel_size=5,
                stride=1, padding=1, bias=False
            ),
            nn.BatchNorm2d(self.channels // 8),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        self.conv23 = nn.Sequential(
            nn.Conv2d(
                in_channels=self.channels//4, out_channels=self.channels//4, kernel_size=3,
                stride=1, padding=0, bias=False
            ),
            nn.BatchNorm2d(self.channels//4),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        self.conv25 = nn.Sequential(
            nn.Conv2d(
                in_channels=self.channels // 4, out_channels=self.channels // 4, kernel_size=5,
                stride=1, padding=1, bias=False
            ),
            nn.BatchNorm2d(self.channels // 4),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        self.conv33 = nn.Sequential(
            nn.Conv2d(
                in_channels=self.channels//2, out_channels=self.channels//2, kernel_size=3,
                stride=1, padding=0, bias=False
            ),
            nn.BatchNorm2d(self.channels//2),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        self.conv35 = nn.Sequential(
            nn.Conv2d(
                in_channels=self.channels//2, out_channels=self.channels//2, kernel_size=5,
                stride=1, padding=1, bias=False
            ),
            nn.BatchNorm2d(self.channels//2),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        self.conv43 = nn.Sequential(
            nn.Conv2d(
                in_channels=self.channels , out_channels=self.channels, kernel_size=3,
                stride=1, padding=0, bias=False
            ),
            nn.BatchNorm2d(self.channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        self.conv45 = nn.Sequential(
            nn.Conv2d(
                in_channels=self.channels, out_channels=self.channels, kernel_size=5,
                stride=1, padding=1, bias=False
            ),
            nn.BatchNorm2d(self.channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        self.linear1 = nn.Sequential(
            nn.Linear(self.channels* 2 * self.init_pixel*self.init_pixel ,1),
            nn.Sigmoid()
        )

    def forward(self, x, y):
        # Convolutional layers
        if print_flag:
            print('d',x.shape,y.shape)
        x = x.reshape(x.shape[0],1,self.pixels_edge,self.pixels_edge)

        # y1 = Variable(torch.randn(y.shape[0],  self.pixels_edge, self.pixels_edge,y.shape[1] ))
        # for i in range(y1.shape[0]):
            # for j in range(y1.shape[1]):
            #     for k in range(y1.shape[2]):
            #         y1[i] = torch.tensor(y[i])
        # y1= torch.tensor(np.stack([np.tile(y[i].cpu(),(self.final_pixel,self.final_pixel,1)) for i in range(y.shape[0])]))
        # # y1 = torch.tensor(np.stack([y[i] for i in range(y.shape[0])]))
        # if torch.cuda.is_available(): y1 = y1.cuda()

        y = y.reshape(y.shape[0], y.shape[3],y.shape[1], y.shape[2])
        if print_flag:
            print('d',x.shape,y.shape)
        #
        x = torch.cat([x, y], 1)
        if print_flag:
            print('d',x.shape)
        #
        x13 = self.conv13(x)
        if print_flag:
            print('d', x13.shape)

        x15 = self.conv15(x)
        if print_flag:
            print('d', x15.shape)

        x = torch.cat([x13, x15], 1)

        x23 = self.conv23(x)
        if print_flag:
            print('d', x23.shape)

        x25 = self.conv25(x)
        if print_flag:
            print('d', x25.shape)

        x = torch.cat([x23, x25], 1)

        x33 = self.conv33(x)
        if print_flag:
            print('d', x33.shape)

        x35 = self.conv35(x)
        if print_flag:
            print('d', x35.shape)

        x = torch.cat([x33, x35], 1)

        x43 = self.conv43(x)
        if print_flag:
            print('d', x43.shape)

        x45 = self.conv45(x)
        if print_flag:
            print('d', x45.shape)

        x = torch.cat([x43, x45], 1)

        x = x.reshape(-1,x.shape[1]*x.shape[2]*x.shape[3])
        if print_flag:
            print('d', x.shape)        #
        x = self.linear1(x)
        if print_flag:
            print('d', x.shape)
        return x




class GeneratorNet(torch.nn.Module):

    def __init__(self, channels,factor, Kernel_factor, z_dim, num_condition, num_pixels):
        self.z_dim = z_dim
        self.num_pixels = num_pixels
        self.pixels_edge = int(np.sqrt(num_pixels))
        self.num_condition = num_condition
        self.channels = channels//factor
        self.init_pixel = 2
        # self.init_pixel = self.pixels_edge

        super(GeneratorNet, self).__init__()

        self.linear1 = nn.Sequential(
            nn.Linear(self.z_dim + self.num_condition, self.init_pixel * self.init_pixel* self.channels),
            nn.BatchNorm1d(self.init_pixel * self.init_pixel* self.channels),
            nn.LeakyReLU(negative_slope=0.2,inplace=True)
        )

        self.deconv13 = nn.Sequential(
            nn.ConvTranspose2d(self.channels, self.channels//2,kernel_size=3,
                stride=1, padding=0, bias = False),
            nn.BatchNorm2d(self.channels//2),
            nn.LeakyReLU(negative_slope=0.2,inplace=True)
        )
        self.deconv15 = nn.Sequential(
            nn.ConvTranspose2d(self.channels, self.channels // 2, kernel_size=5,
                               stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channels // 2),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        self.deconv23 = nn.Sequential(
            nn.ConvTranspose2d(self.channels, self.channels // 4, kernel_size=3,
                               stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.channels//4),
            nn.LeakyReLU(negative_slope=0.2,inplace=True)
        )
        self.deconv25 = nn.Sequential(
            nn.ConvTranspose2d(self.channels, self.channels // 4, kernel_size=5,
                               stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channels // 4),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        self.deconv33 = nn.Sequential(
            nn.ConvTranspose2d(self.channels // 2, self.channels // 8, kernel_size=3,
                               stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.channels // 8),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        self.deconv35 = nn.Sequential(
            nn.ConvTranspose2d(self.channels // 2, self.channels // 8, kernel_size=5,
                               stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channels // 8),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        self.deconv4 = nn.Sequential(
            nn.ConvTranspose2d(self.channels//4, 1, kernel_size=3,
                               stride=1, padding=0, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )



    def forward(self, x, y):
        if print_flag:
            print('g',x.shape,y.shape)

        x = x.reshape(x.size(0), x.size(1))
        if print_flag:
            print('g', x.shape, y.shape)

        x = torch.cat([x, y], 1)
        if print_flag:
            print('g', x.shape)

        #Linear Layer
        x = self.linear1(x)
        if print_flag:
            print('g', x.shape)

        x = x.reshape(-1,self.channels,self.init_pixel, self.init_pixel)
        if print_flag:
            print('g', x.shape)
        # Convolutional layers
        x13 = self.deconv13(x)
        if print_flag:
            print('g', x13.shape)
        x15 = self.deconv15(x)
        if print_flag:
            print('g', x15.shape)

        x = torch.cat([x13, x15], 1)
        if print_flag:
            print('g', x.shape)

        x23 = self.deconv23(x)
        if print_flag:
            print('g', x23.shape)
        x25 = self.deconv25(x)
        if print_flag:
            print('g', x25.shape)

        x = torch.cat([x23, x25], 1)
        if print_flag:
            print('g', x.shape)
        x33 = self.deconv33(x)
        if print_flag:
            print('g', x33.shape)
        x35 = self.deconv35(x)
        if print_flag:
            print('g', x35.shape)

        x = torch.cat([x33, x35], 1)
        if print_flag:
            print('g', x.shape)
        x = self.deconv4(x)
        if print_flag:
            print('g', x.shape)
        return x


#
def initialize_models(channels,factor, Kernel_factor,z_dim, num_condition, num_pixels):
    discriminator = DiscriminatorNet(channels,factor, Kernel_factor,z_dim, num_condition,num_pixels)
    generator = GeneratorNet(channels,factor, Kernel_factor,z_dim, num_condition, num_pixels)

    if torch.cuda.is_available():
        discriminator.cuda()
        generator.cuda()

    # Optimizers
    d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002)
    g_optimizer = optim.Adam(generator.parameters(), lr=0.0002)

    # Loss function (Creates a criterion that measures the Binary Cross Entropy
    # between the target and the output)
    loss = nn.BCELoss()
    # loss_g = LossBGANConGen()
    # loss_d = LossBGANConDis()
    return generator, discriminator, g_optimizer, d_optimizer, loss

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