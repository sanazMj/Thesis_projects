
'Works for Partial images on 9*9 EES dataset'
'Code for publication crc-cgan'
import torch
from torch import nn, optim
import numpy as np
from torch.autograd.variable import Variable
from Minibatch_discrimination import *
# from Loss import LossBGANConGen, LossBGANConDis
Print = False
class DiscriminatorNet(torch.nn.Module):

    def __init__(self,channels,z_dim,num_condition,num_pixels, Minibatch,PacGAN_pacnum, mode_collapse, gen_num):
        super(DiscriminatorNet, self).__init__()
        self.minibatch = Minibatch
        self.z_dim = num_pixels
        self.num_condition = 0
        self.gen_num = gen_num
        self.n_out = 1 + self.gen_num
        self.mode_collapse = mode_collapse
        self.last_opt = nn.Sigmoid() if gen_num ==0 else nn.Softmax()
        self.num_pixels = num_pixels
        self.pixels_edge = int(np.sqrt(num_pixels))
        self.channels = channels
        self.init_pixel = 5
        self.Packnum = PacGAN_pacnum
        self.final_pixel = self.pixels_edge
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=(1 + self.num_condition)*self.Packnum , out_channels=channels//8, kernel_size=3,
                stride=1, padding=1, bias=False
            ),
            nn.BatchNorm2d(channels//8),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=channels//8, out_channels=channels//4, kernel_size=3,
                stride=1, padding=1, bias=False
            ),
            nn.BatchNorm2d(channels//4),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        # self.conv3 = nn.Sequential(
        #     nn.Conv2d(
        #         in_channels=channels//4, out_channels=channels//2, kernel_size=3,
        #         stride=1, padding=0, bias=False
        #     ),
        #     nn.BatchNorm2d(channels//2),
        #     nn.LeakyReLU(negative_slope=0.2, inplace=True)
        # )
        if Minibatch:
            self.linear1 = nn.Sequential(
                nn.Linear(channels // 2 * self.init_pixel * self.init_pixel, self.n_out ),
            )
        else:
            self.linear1 = nn.Sequential(
                nn.Linear(channels//4* self.init_pixel*self.init_pixel ,self.n_out ),
            )

    def forward(self, x, y):
        # Convolutional layers
        if Print:
            print('d',x.shape,y.shape)
        x = x.reshape(int(x.shape[0]), 1*self.Packnum , self.pixels_edge, self.pixels_edge)
        if Print:
            print('d',x.shape,y.shape)
        # y1 = Variable(torch.randn(y.shape[0],  self.pixels_edge, self.pixels_edge,y.shape[1] ))
        # for i in range(y1.shape[0]):
            # for j in range(y1.shape[1]):
            #     for k in range(y1.shape[2]):
            #         y1[i] = torch.tensor(y[i])
        # y1= torch.tensor(np.stack([np.tile(y[i].cpu(),(self.final_pixel,self.final_pixel,1)) for i in range(y.shape[0])]))
        # # y1 = torch.tensor(np.stack([y[i] for i in range(y.shape[0])]))
        # if torch.cuda.is_available(): y1 = y1.cuda()

        # y = y.reshape(y.shape[0], y.shape[3],y.shape[1], y.shape[2])
        # if Print:
        #     print('d', x.shape, y.shape)
        # x = torch.cat([x, y], 1)
        if Print:
            print('d', x.shape)        #
        x = self.conv1(x)
        if Print:
            print('d', x.shape)
        x = self.conv2(x)
        if Print:
            print('d', x.shape)
            # x = self.conv3(x)
        # print('d',x.shape)

        x1 = x.reshape(-1,x.shape[1]*x.shape[2]*x.shape[3])
        if Print:
            print('d', x1.shape)
        if self.minibatch:
            x1 = self.minibatch(x1)
        if Print:
            print('d', x1.shape)
            #
        x = self.linear1(x1)
        if Print:
            print('d', x.shape)

        x = self.last_opt(x)

        if self.mode_collapse == 'GDPP':
            return x, x1
        else:
            return x




class GeneratorNet(torch.nn.Module):

    def __init__(self, channels, z_dim, num_condition, num_pixels):
        self.pixels_edge = int(np.sqrt(num_pixels))
        self.z_dim = 100
        self.num_pixels = num_pixels
        self.pixels_edge = int(np.sqrt(num_pixels))
        self.num_condition = 0
        self.channels = channels
        self.init_pixel = 5
        # self.init_pixel = self.pixels_edge

        super(GeneratorNet, self).__init__()

        self.linear1 = nn.Sequential(
            nn.Linear(self.z_dim + self.num_condition, self.init_pixel * self.init_pixel* self.channels//4),
            nn.BatchNorm1d(self.init_pixel * self.init_pixel* self.channels//4),
            nn.LeakyReLU(negative_slope=0.2,inplace=True)
        )

        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(channels//4, channels//8,kernel_size=3,
                stride=1, padding=1, bias = False),
            nn.BatchNorm2d(channels//8),
            nn.LeakyReLU(negative_slope=0.2,inplace=True)
        )
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(channels//8, 1, kernel_size=3,
                               stride=1, padding=1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        # self.deconv3 = nn.Sequential(
        #     nn.ConvTranspose2d(channels//4, 1, kernel_size=3,
        #                        stride=1, padding=0, bias=False),
        #     nn.BatchNorm2d(1),
        #     nn.Sigmoid()
        # )


    def forward(self, x, y):
        if Print:
            print('g',x.shape)

        x = x.reshape(x.size(0), x.size(1))
        # print('g', x.shape, y.shape)
        if Print:
            print('g', x.shape)
        # x = torch.cat([x, y], 1)
        # print('g',x.shape)
        # if Print:
        #     print('g', x.shape)
        #Linear Layer
        x = self.linear1(x)
        # print('g',x.shape)
        if Print:
            print('g', x.shape)
        x = x.reshape(-1,self.channels//4,self.init_pixel, self.init_pixel)
        # print('g', x.shape)
        if Print:
            print('g', x.shape)
        # Convolutional layers
        x = self.deconv1(x)
        # print('g',x.shape)
        if Print:
            print('g', x.shape)
        x = self.deconv2(x)
        # print('g',x.shape)
        #
        if Print:
            print('g', x.shape)
        # x = self.deconv3(x)
        # print('g',x.shape)

        return x

class VARNET(torch.nn.Module):

    def __init__(self,channels,z_dim,num_condition,num_pixels, Minibatch=False, Packnum=1):
        super(VARNET, self).__init__()
        self.minibatch = Minibatch
        self.z_dim = num_pixels
        self.num_condition = 0
        self.num_pixels = num_pixels
        self.pixels_edge = int(np.sqrt(num_pixels))
        self.channels = channels
        self.init_pixel = 5
        self.Packnum = Packnum
        self.final_pixel = self.pixels_edge
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=(1 + self.num_condition)*self.Packnum , out_channels=channels//8, kernel_size=3,
                stride=1, padding=1, bias=False
            ),
            nn.BatchNorm2d(channels//8),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=channels//8, out_channels=channels//4, kernel_size=3,
                stride=1, padding=1, bias=False
            ),
            nn.BatchNorm2d(channels//4),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        # self.conv3 = nn.Sequential(
        #     nn.Conv2d(
        #         in_channels=channels//4, out_channels=channels//2, kernel_size=3,
        #         stride=1, padding=0, bias=False
        #     ),
        #     nn.BatchNorm2d(channels//2),
        #     nn.LeakyReLU(negative_slope=0.2, inplace=True)
        # )
        if Minibatch:
            self.linear1 = nn.Sequential(
                nn.Linear(channels // 2 * self.init_pixel * self.init_pixel, 1),
                nn.Sigmoid()
            )
        else:
            self.linear1 = nn.Sequential(
                nn.Linear(channels//4* self.init_pixel*self.init_pixel ,1),
                nn.Sigmoid()
            )

    def forward(self, x, y):
        # Convolutional layers
        if Print:
            print('d',x.shape,y.shape)
        x = x.reshape(int(x.shape[0]), 1*self.Packnum , self.pixels_edge, self.pixels_edge)
        if Print:
            print('d',x.shape,y.shape)
        # y1 = Variable(torch.randn(y.shape[0],  self.pixels_edge, self.pixels_edge,y.shape[1] ))
        # for i in range(y1.shape[0]):
            # for j in range(y1.shape[1]):
            #     for k in range(y1.shape[2]):
            #         y1[i] = torch.tensor(y[i])
        # y1= torch.tensor(np.stack([np.tile(y[i].cpu(),(self.final_pixel,self.final_pixel,1)) for i in range(y.shape[0])]))
        # # y1 = torch.tensor(np.stack([y[i] for i in range(y.shape[0])]))
        # if torch.cuda.is_available(): y1 = y1.cuda()

        # y = y.reshape(y.shape[0], y.shape[3],y.shape[1], y.shape[2])
        # if Print:
        #     print('d', x.shape, y.shape)
        # x = torch.cat([x, y], 1)
        if Print:
            print('d', x.shape)        #
        x = self.conv1(x)
        if Print:
            print('d', x.shape)
        x = self.conv2(x)
        if Print:
            print('d', x.shape)
            # x = self.conv3(x)
        # print('d',x.shape)

        x = x.reshape(-1,x.shape[1]*x.shape[2]*x.shape[3])
        if Print:
            print('d', x.shape)
        if self.minibatch:
            x = self.minibatch(x)
        if Print:
            print('d', x.shape)
            #
        x = self.linear1(x)
        if Print:
            print('d', x.shape)

        return x




#
def initialize_models(channels,channel_factor, kernel_factor, z_dim, num_condition, num_pixels,Minibatch, minibatch_kind, Packnum,PacGAN_pacnum, mode_collapse, gen_num=0):
    var_optimizer = []
    varnet = []

    pixels_edge = int(np.sqrt(num_pixels))
    if Minibatch:
        minibatch_net = MinibatchDiscrimination(channels//4 * pixels_edge * pixels_edge, channels//4 *pixels_edge * pixels_edge,5, minibatch_kind)
    else:
        minibatch_net = False

    discriminator = DiscriminatorNet(channels,z_dim, num_condition,num_pixels,minibatch_net, PacGAN_pacnum, mode_collapse, gen_num)
    generator = []
    g_optimizer = []
    if gen_num !=0 :
        for gn in range(gen_num):
            generator_temp = GeneratorNet(channels,z_dim, num_condition, num_pixels)
            if torch.cuda.is_available():
                generator_temp.cuda()
            generator.append(generator_temp)
            g_optimizer_temp = optim.Adam(generator_temp.parameters(), lr=0.0002)
            g_optimizer.append(g_optimizer_temp)

    else:
        generator = GeneratorNet(channels, z_dim, num_condition, num_pixels)
        generator.cuda()

    if mode_collapse == 'VarGAN':
        varnet = VARNET(channels,z_dim, num_condition,num_pixels,minibatch_net, Packnum)

    if torch.cuda.is_available():
        discriminator.cuda()
        if mode_collapse == 'VarGAN':
            varnet.cuda()
    # Optimizers
    d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002)
    if mode_collapse == 'VarGAN':
        var_optimizer = optim.Adam(varnet.parameters(), lr=0.0002)

    # Loss function (Creates a criterion that measures the Binary Cross Entropy
    # between the target and the output)
    loss = nn.BCELoss()
    # loss_g = LossBGANConGen()
    # loss_d = LossBGANConDis()
    # print(generator, discriminator, varnet, g_optimizer, d_optimizer, var_optimizer, loss)
    return generator, discriminator, varnet, g_optimizer, d_optimizer, var_optimizer, loss

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