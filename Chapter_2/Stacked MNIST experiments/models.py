import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd.variable import Variable
import torch.nn.functional as F
import torch as T
from torchsummary import summary

import numpy as np
#### VAnilla GAN From mode collapse model for MNIST
import torch
from torch import nn, optim
Print_flag = False

def normal_init(m, mean, std):
    if isinstance(m, nn.Linear):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()

class vanilla_discriminator(torch.nn.Module):
    """
    A three hidden-layer discriminative neural network
    """

    def __init__(self, num_pixel, pac_dim, Stacked=True,  minibatch_d=False):
        super(vanilla_discriminator, self).__init__()
        n_out = 1
        self.num_pixel = num_pixel
        self.Stacked = Stacked
        self.minibatch_layer = minibatch_d
        self.pac_dim = pac_dim

        if self.Stacked:
            self.num_pixel = self.num_pixel * 3
        self.hidden0 = nn.Sequential(
            nn.Linear(self.num_pixel * self.pac_dim , 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )
        self.hidden1 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )
        self.hidden2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )
        if self.minibatch_layer:
            self.out = nn.Sequential(
                torch.nn.Linear(128 + 256, n_out),
                torch.nn.Sigmoid()
            )
        else:
            self.out = nn.Sequential(
                torch.nn.Linear(256, n_out),
                torch.nn.Sigmoid()
            )
        # self.out = nn.Sequential(
        #     torch.nn.Linear(256, n_out),
        #     torch.nn.Sigmoid()
        # )
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)
    def forward(self, x):
        if Print_flag:
            print('d', x.shape)
        if self.Stacked:
            # print(self.num_pixel,self.pac_dim)
            x = x.reshape(-1,self.num_pixel * self.pac_dim)
            if Print_flag:
                print('d', x.shape)
        # x = torch.cat((x, y), 1)
        x = self.hidden0(x)
        if Print_flag:
            print('d',x.shape)
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


class vanilla_varnet(torch.nn.Module):
    """
    A three hidden-layer discriminative neural network
    """

    def __init__(self, num_pixel, pac_var, Stacked=True,  minibatch_d=False):
        super(vanilla_varnet, self).__init__()
        n_out = 1
        self.num_pixel = num_pixel
        self.Stacked = Stacked
        self.minibatch_layer = minibatch_d
        self.pac_var = pac_var

        if self.Stacked:
            self.num_pixel = self.num_pixel * 3
        self.hidden0 = nn.Sequential(
            nn.Linear(self.num_pixel * self.pac_var , 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )
        self.hidden1 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )
        self.hidden2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )
        if self.minibatch_layer:
            self.out = nn.Sequential(
                torch.nn.Linear(128 + 256, n_out),
                torch.nn.Sigmoid()
            )
        else:
            self.out = nn.Sequential(
                torch.nn.Linear(256, n_out),
                torch.nn.Sigmoid()
            )
        # self.out = nn.Sequential(
        #     torch.nn.Linear(256, n_out),
        #     torch.nn.Sigmoid()
        # )
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)
    def forward(self, x):
        if Print_flag:
            print('v', x.shape)
        if self.Stacked:
            x = x.reshape(-1,self.num_pixel * self.pac_var)
            if Print_flag:
                print('v', x.shape)
        # x = torch.cat((x, y), 1)
        x = self.hidden0(x)
        if Print_flag:
            print('v',x.shape)
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

class vanilla_generator(torch.nn.Module):
    """
    A three hidden-layer generative neural network
    """

    def __init__(self, num_condition, num_pixels, n_features, Stacked=True):
        super(vanilla_generator, self).__init__()
        self.n_features = n_features
        self.num_pixels = num_pixels
        n_out = num_pixels
        self.Stacked = Stacked
        if self.Stacked:
            n_out = num_pixels * 3
        self.hidden0 = nn.Sequential(
            nn.Linear(n_features, 256),
            nn.LeakyReLU(0.2)
        )
        self.hidden1 = nn.Sequential(
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2)
        )
        self.hidden2 = nn.Sequential(
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2)
        )

        self.out = nn.Sequential(
            nn.Linear(1024, n_out),
            nn.Tanh()
        )
        # if self.Binary and not self.Stacked:
        #     self.out = nn.Sequential(
        #         nn.Linear(1024, n_out),
        #         nn.Sigmoid()
        #     )
        # else:
        #     self.out = nn.Sequential(
        #         nn.Linear(1024, n_out),
        #         nn.Tanh()
        #     )
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)
    def forward(self, x):
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
        if self.Stacked:
            x = x.reshape (-1, 3, int(np.sqrt(self.num_pixels)), int(np.sqrt(self.num_pixels)) )
            if Print_flag:
                print('g', x.shape)
        return x

#
# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv2d(1, 32, 3, 1)
#         self.conv2 = nn.Conv2d(32, 64, 3, 1)
#         self.dropout1 = nn.Dropout2d(0.25)
#         self.dropout2 = nn.Dropout2d(0.5)
#         self.fc1 = nn.Linear(12544, 128)
#
#         # self.fc1 = nn.Linear(64 * 12 * 12, 128)
#         self.fc2 = nn.Linear(128, 10)
#
#     def forward(self, x):
#         print(x.shape)
#         x = self.conv1(x)
#         print(x.shape)
#
#         x = F.relu(x)
#         print(x.shape)
#         x = self.conv2(x)
#         print(x.shape)
#         x = F.relu(x)
#         print(x.shape)
#
#         x = F.max_pool2d(x, 2)
#         print(x.shape)
#
#         x = self.dropout1(x)
#         print(x.shape)
#
#         x = torch.flatten(x, 1)
#         print(x.shape)
#
#         x = self.fc1(x)
#         print(x.shape)
#
#         x = F.relu(x)
#         print(x.shape)
#
#         x = self.dropout2(x)
#         print(x.shape)
#
#         x = self.fc2(x)
#         print(x.shape)
#
#         output = F.log_softmax(x, dim=1)
#         print(x.shape)
#
#         return output
#
#
#
def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()

############################################################
######################  DCGAN ##############################
############################################################
############################################################
class generator_net(nn.Module):
    # initializers
    def __init__(self, feature, d=128, c_dim=3, Binary=False, Stacked=False, Veegan=False, img_size=32):
        self.d = d
        self.c_dim = c_dim
        self.Binary = Binary
        self.Stacked = Stacked
        self.img_size = img_size
        self.feature = feature
        super(generator_net, self).__init__()
        if Veegan == False:
            self.deconv1_1 = nn.ConvTranspose2d(self.feature, self.d*2, 4, 1, 0)
        else:
            self.deconv1_1 = nn.ConvTranspose2d(self.img_size*self.img_size, self.d*2, 4, 1, 0)

        self.deconv1_1_bn = nn.BatchNorm2d(self.d*2)
        # self.deconv1_2 = nn.ConvTranspose2d(10, self.d*2, 4, 1, 0)
        # self.deconv1_2_bn = nn.BatchNorm2d(self.d*2)
        self.deconv2 = nn.ConvTranspose2d(self.d*2, self.d*1, 4, 2, 1)
        self.deconv2_bn = nn.BatchNorm2d(self.d*1)
        self.deconv3 = nn.ConvTranspose2d(self.d*1, int(self.d/2), 4, 2, 1)
        self.deconv3_bn = nn.BatchNorm2d(int(self.d/2))
        self.deconv4 = nn.ConvTranspose2d(int(self.d/2), self.c_dim , 4, 2, 1)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input):
        x = input.view(-1,input.shape[1], 1, 1)
        x = F.relu(self.deconv1_1_bn(self.deconv1_1(x)))
        # y = F.relu(self.deconv1_2_bn(self.deconv1_2(label)))
        # x = torch.cat([x, y], 1)
        x = F.relu(self.deconv2_bn(self.deconv2(x)))
        x = F.relu(self.deconv3_bn(self.deconv3(x)))
        if self.Binary and not self.Stacked:
            x = torch.sigmoid(self.deconv4(x))
        else:
            x = torch.tanh(self.deconv4(x))
        # x = F.relu(self.deconv4_bn(self.deconv4(x)))
        # x = F.tanh(self.deconv5(x))

        return x

class varnet_net(nn.Module):
    # initializers
    def __init__(self, pac_dim, d=128, c_dim=3, minibatch_d=False, Veegan=False, img_size=32):
        # '''
        # c_dim  = num_channels
        # pack_dim = number of samples packed and passed to the discriminator
        # For instance: each image: 28x28x3 pack_dim = 4 -> discriminator input 28x28x12
        # '''
        self.d = d
        self.c_dim = c_dim
        self.pack_dim = pac_dim
        self.Veegan = Veegan
        self.img_size = img_size
        super(varnet_net, self).__init__()
        self.minibatch_layer = minibatch_d
        self.conv1_1 = nn.Conv2d(self.c_dim * self.pack_dim, int(self.d/2), 4, 2, 1)
        if self.Veegan == True:
            self.conv1_2 = nn.Conv2d(1, int(self.d/2), 4, 2, 1)

        self.conv2 = nn.Conv2d(int(self.d/2) * (int(self.Veegan)+1), self.d * (int(self.Veegan)+1), 4, 2, 1)
        self.conv2_bn = nn.BatchNorm2d(self.d * (int(self.Veegan)+1))
        self.conv3 = nn.Conv2d(self.d* (int(self.Veegan)+1), self.d*2* (int(self.Veegan)+1), 4, 2, 1)
        self.conv3_bn = nn.BatchNorm2d(self.d*2* (int(self.Veegan)+1))

        if self.minibatch_layer == False:
            self.conv4 = nn.Conv2d(self.d *2 * (int(self.Veegan)+1), 1, 4, 1, 0)
        else:
            self.conv4 = nn.Conv2d(self.d *3, 1, 4, 1, 0)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input):
        batch_size_x = input.shape[0]
        # print(input.shape, z_recon.shape)
        x = F.leaky_relu(self.conv1_1(input), 0.2)
        # print(x.shape)
        # if self.Veegan:
        #     z_recon = z_recon.reshape(-1, 1, self.img_size,self.img_size)
        #     z_recon = F.leaky_relu(self.conv1_2(z_recon), 0.2)
        #     # print(x.shape, z_recon.shape)
        #
        #     x = torch.cat([x, z_recon], 1)
        #     # print(x.shape, z_recon.shape)

        x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)

        if self.minibatch_layer:
            x = x.reshape(-1, self.d * 2 * 4 * 4)
            # print('d before minibatch',x.shape)
            x = self.minibatch_layer(x)
            # print('d after minibatch',x.shape)
            x = x.reshape(batch_size_x, -1, 4, 4)
        x = torch.sigmoid(self.conv4(x))

        return x


class discriminator_net(nn.Module):
    # initializers
    def __init__(self, pac_dim, d=128, c_dim=3, minibatch_d=False, Veegan=False, img_size=32):
        # '''
        # c_dim  = num_channels
        # pack_dim = number of samples packed and passed to the discriminator
        # For instance: each image: 28x28x3 pack_dim = 4 -> discriminator input 28x28x12
        # '''
        self.d = d
        self.c_dim = c_dim
        self.pack_dim = pac_dim
        self.Veegan = Veegan
        self.img_size = img_size
        super(discriminator_net, self).__init__()
        self.minibatch_layer = minibatch_d
        self.conv1_1 = nn.Conv2d(self.c_dim * self.pack_dim, int(self.d/2), 4, 2, 1)
        if self.Veegan == True:
            self.conv1_2 = nn.Conv2d(1, int(self.d/2), 4, 2, 1)

        self.conv2 = nn.Conv2d(int(self.d/2) * (int(self.Veegan)+1), self.d * (int(self.Veegan)+1), 4, 2, 1)
        self.conv2_bn = nn.BatchNorm2d(self.d * (int(self.Veegan)+1))
        self.conv3 = nn.Conv2d(self.d* (int(self.Veegan)+1), self.d*2* (int(self.Veegan)+1), 4, 2, 1)
        self.conv3_bn = nn.BatchNorm2d(self.d*2* (int(self.Veegan)+1))

        if self.minibatch_layer == False:
            self.conv4 = nn.Conv2d(self.d *2 * (int(self.Veegan)+1), 1, 4, 1, 0)
        else:
            self.conv4 = nn.Conv2d(self.d *3, 1, 4, 1, 0)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input):
        batch_size_x = input.shape[0]
        # print(input.shape, z_recon.shape)
        x = F.leaky_relu(self.conv1_1(input), 0.2)
        # print(x.shape)
        # if self.Veegan:
        #     z_recon = z_recon.reshape(-1, 1, self.img_size,self.img_size)
        #     z_recon = F.leaky_relu(self.conv1_2(z_recon), 0.2)
        #     # print(x.shape, z_recon.shape)
        #
        #     x = torch.cat([x, z_recon], 1)
        #     # print(x.shape, z_recon.shape)

        x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)

        if self.minibatch_layer:
            x = x.reshape(-1, self.d * 2 * 4 * 4)
            # print('d before minibatch',x.shape)
            x = self.minibatch_layer(x)
            # print('d after minibatch',x.shape)
            x = x.reshape(batch_size_x, -1, 4, 4)
        x = torch.sigmoid(self.conv4(x))

        return x



class Generator(nn.Module):
    def __init__(self, n_features):
        super(Generator, self).__init__()
        self.features = n_features
        self.Linear1 = nn.Sequential(
            nn.Linear(self.features, 2 * 2 * 512),
            nn.BatchNorm1d(2 * 2 * 512),
            nn.ReLU()
        )
        self.Conv_trans1 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=5, padding=2, stride=3),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.Conv_trans2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=5, padding=2, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.Conv_trans3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, padding=1, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.Conv_trans4 = nn.Sequential(
            nn.ConvTranspose2d(64, 3, kernel_size=4, padding=1, stride=2),
            nn.Tanh()
        )

    def forward(self, x):
        # print(x.shape)
        x = self.Linear1(x)
        # print(x.shape)
        x = x.reshape(-1, 512, 2, 2)
        # print(x.shape)
        x = self.Conv_trans1(x)
        # print(x.shape)
        x = self.Conv_trans2(x)
        # print(x.shape)
        x = self.Conv_trans3(x)
        # print(x.sh/ape)
        x = self.Conv_trans4(x)
        # print(x.shape)
        return x

    # %%


class Discriminator(nn.Module):
    def __init__(self, pac_dim):
        super(Discriminator, self).__init__()
        self.pac_dim = pac_dim
        self.Conv1 = nn.Sequential(
            nn.Conv2d(3 * self.pac_dim, 64, kernel_size=5, padding=2, stride=2),
            nn.LeakyReLU()
        )
        self.Conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=5, padding=2, stride=2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU()
        )
        self.Conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=5, padding=2, stride=2),
            nn.BatchNorm2d(256),
            nn.LeakyReLU()
        )
        self.Conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=5, padding=2, stride=2),
            nn.LeakyReLU()
        )
        self.Linear1 = nn.Sequential(
            nn.Linear(2 * 2 * 512, 1),
            nn.BatchNorm1d(1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # print(x.shape)
        x = self.Conv1(x)
        # print(x.shape)
        x = self.Conv2(x)
        # print(x.shape)

        x = self.Conv3(x)
        # print(x.shape)
        x = self.Conv4(x)
        # print(x.shape)
        x = x.reshape(-1, 512 * 2 * 2)
        # print(x.shape)
        x = self.Linear1(x)
        # print(x.shape)
        return x

    # %%


class VARNET(nn.Module):
    def __init__(self, pack_num):
        super(VARNET, self).__init__()
        self.pack_num = pack_num
        self.Conv1 = nn.Sequential(
            nn.Conv2d(3 * self.pack_num, 64, kernel_size=5, padding=2, stride=2),
            nn.LeakyReLU()
        )
        self.Conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=5, padding=2, stride=2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU()
        )
        self.Conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=5, padding=2, stride=2),
            nn.BatchNorm2d(256),
            nn.LeakyReLU()
        )
        self.Conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=5, padding=2, stride=2),
            nn.LeakyReLU()
        )
        self.Linear1 = nn.Sequential(
            nn.Linear(2 * 2 * 512, 1),
            nn.BatchNorm1d(1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # print(x.shape)
        x = self.Conv1(x)
        # print(x.shape)
        x = self.Conv2(x)
        # print(x./shape)

        x = self.Conv3(x)
        # print(x.shape)
        x = self.Conv4(x)
        # print(x.shape)
        x = x.reshape(-1, 512 * 2 * 2)
        # print(x.shape)
        x = self.Linear1(x)
        # print(x.shape)
        return x

#######################3
### pacGAN=DCGAN
#####
class Generator_pacGAN(nn.Module):
    # initializers
    def __init__(self, features, d=128):
        super(Generator_pacGAN, self).__init__()
        self.features = features
        self.d = d
        self.linear1 =  nn.Sequential(nn.Linear(100, d*4* 4*4),
                                      nn.BatchNorm1d(d*4*4*4),
                                      nn.ReLU())
        self.deconv1 = nn.Sequential(nn.ConvTranspose2d(100, d*4, 4, 1, 0),
                                    nn.BatchNorm2d(d*4),
                                     nn.ReLU())
        self.deconv2 = nn.Sequential(nn.ConvTranspose2d(d*4, d*2, 4, 2, 1),
                                    nn.BatchNorm2d(d*2),
                                     nn.ReLU())
        self.deconv3 = nn.Sequential(nn.ConvTranspose2d(d*2, d*1, 4, 2, 1),
                                    nn.BatchNorm2d(d*1),
                                     nn.ReLU())
        self.deconv4 = nn.Sequential(nn.ConvTranspose2d(d*1, d//2, 4, 2, 1),
                                    nn.BatchNorm2d(d//2),
                                     nn.ReLU())
        self.deconv5 = nn.Sequential(nn.ConvTranspose2d(d//2, 3, 4, 2, 1),
                                    nn.Tanh())
    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input):
        # print(input.shape)
        # x = input.reshape(-1,100,1,1)

        x = input.reshape(-1,100)
        # print(x.shape)
        # x = self.deconv1(x)
        # print(x.shape)
        x = self.linear1(x)
        x = x.reshape(-1,self.d * 4, 4, 4)
        # print(x.shape)

        x = self.deconv2(x)
        x = self.deconv3(x)
        x = self.deconv4(x)
        x = self.deconv5(x)

        return x

class Discriminator_pacGAN(nn.Module):
    # initializers
    def __init__(self, pac_dim, d=128):
        super(Discriminator_pacGAN, self).__init__()
        self.d = d
        self.conv1 = nn.Sequential(
                        nn.Conv2d(3 * pac_dim, d//2, 4, 2, 1),
                        nn.LeakyReLU(0.2)
                    )
        self.conv2 = nn.Sequential(nn.Conv2d(d//2, d, 4, 2, 1),
                                   nn.BatchNorm2d(d),
                                   nn.LeakyReLU(0.2)
                                   )
        self.conv3 = nn.Sequential(nn.Conv2d(d, d*2, 4, 2, 1),
                                    nn.BatchNorm2d(d*2),
                                   nn.LeakyReLU(0.2)
                                   )
        self.conv4 =  nn.Sequential(nn.Conv2d(d*2, d*4, 4, 2, 1),
                                  nn.BatchNorm2d(d*4),
                                    nn.LeakyReLU(0.2)
                                    )
        self.conv5 =  nn.Sequential(nn.Conv2d(d*4, 1, 4, 1, 0)
                                ,nn.Sigmoid())
        self.linear1 = nn.Sequential(nn.Linear(d*4*4*4,1)
                                ,nn.Sigmoid())
    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input):
        x = self.conv1(input)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        # print(x.shape , 'd')
        x = x.reshape(-1, self.d * 4*4*4)
        # print(x.shape , 'd')
        x = self.linear1(x)
        # x = self.conv5(x)

        return x


def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()

# class Generator_pacGAN(nn.Module):
#     def __init__(self, n_features):
#         super(Generator_pacGAN, self).__init__()
#         self.features = n_features
#         self.Linear1 = nn.Sequential(
#             nn.Linear(self.features, 4 * 4 * 1024),
#             nn.BatchNorm1d(4 * 4 * 1024),
#             nn.ReLU()
#         )
#         self.Conv_trans1 = nn.Sequential(
#             nn.ConvTranspose2d(1024, 512, kernel_size=5, stride=2),
#             nn.BatchNorm2d(512),
#             nn.ReLU()
#         )
#         self.Conv_trans2 = nn.Sequential(
#             nn.ConvTranspose2d(512, 256, kernel_size=5, stride=2),
#             nn.BatchNorm2d(256),
#             nn.ReLU()
#         )
#         self.Conv_trans3 = nn.Sequential(
#             nn.ConvTranspose2d(256, 128, kernel_size=5, stride=2),
#             nn.BatchNorm2d(128),
#             nn.ReLU()
#         )
#         self.Conv_trans4 = nn.Sequential(
#             nn.ConvTranspose2d(128, 3, kernel_size=5, stride=2),
#             nn.Tanh()
#         )
#
#     def forward(self, x):
#         print(x.shape)
#         x = self.Linear1(x)
#         print(x.shape)
#         x =  x.reshape(-1,  1024, 4, 4)
#         x = self.Conv_trans1(x)
#         print(x.shape)
#
#         x = self.Conv_trans2(x)
#         print(x.shape)
#         x = self.Conv_trans3(x)
#         print(x.shape)
#         x = self.Conv_trans4(x)
#         print(x.shape)
#         return x
#
# class Discriminator_pacGAN(nn.Module):
#     def __init__(self,pac_dim):
#         super(Discriminator_pacGAN, self).__init__()
#         self.pac_dim = pac_dim
#
#         self.Conv1 = nn.Sequential(
#             nn.Conv2d(3*self.pac_dim, 64, kernel_size=3),
#             nn.BatchNorm2d(64),
#             nn.LeakyReLU()
#         )
#         self.Conv2 = nn.Sequential(
#             nn.Conv2d(64, 128, kernel_size=3),
#             nn.BatchNorm2d(128),
#             nn.LeakyReLU()
#         )
#         self.Conv3 = nn.Sequential(
#             nn.Conv2d(128, 256, kernel_size=3),
#             nn.BatchNorm2d(256),
#             nn.LeakyReLU()
#         )
#         self.Conv4 = nn.Sequential(
#             nn.Conv2d(256, 512, kernel_size=3),
#             nn.BatchNorm2d(512),
#             nn.LeakyReLU()
#         )
#         self.Linear1 = nn.Sequential(
#             nn.Linear(4* 4 * 512, 1),
#             nn.BatchNorm1d(1),
#             nn.Sigmoid()
#         )
#
#     def forward(self, x):
#
#         print(x.shape)
#         x = self.Conv1(x)
#         print(x.shape)
#
#         x = self.Conv2(x)
#         print(x.shape)
#         x = self.Conv3(x)
#         print(x.shape)
#         x = self.Conv4(x)
#         print(x.shape)
#         x = x.reshape(-1, 4*4*512)
#         x = self.Linear1(x)
#         print(x.shape)
#
#         return x

class VARNET_pacGAN(nn.Module):
    # initializers
    def __init__(self, pac_dim, d=128):
        super(VARNET_pacGAN, self).__init__()
        self.d = d
        self.conv1 = nn.Sequential(
                        nn.Conv2d(3 * pac_dim, d//2, 4, 2, 1),
                        nn.LeakyReLU(0.2)
                    )
        self.conv2 = nn.Sequential(nn.Conv2d(d//2, d, 4, 2, 1),
                                   nn.BatchNorm2d(d),
                                   nn.LeakyReLU(0.2)
                                   )
        self.conv3 = nn.Sequential(nn.Conv2d(d, d*2, 4, 2, 1),
                                    nn.BatchNorm2d(d*2),
                                   nn.LeakyReLU(0.2)
                                   )
        self.conv4 =  nn.Sequential(nn.Conv2d(d*2, d*4, 4, 2, 1),
                                  nn.BatchNorm2d(d*4),
                                    nn.LeakyReLU(0.2)
                                    )
        self.conv5 =  nn.Sequential(nn.Conv2d(d*4, 1, 4, 1, 0)
                                ,nn.Sigmoid())
        self.linear1 = nn.Sequential(nn.Linear(d * 4 * 4 * 4,1)
                                     , nn.Sigmoid())
    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input):
        x = self.conv1(input)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        # x = self.conv5(x)
        x = x.reshape(-1, self.d * 4 *4*4)
        x = self.linear1(x)
        # print(x.shape)
        return x
# class VARNET_pacGAN(nn.Module):
#     def __init__(self, pac_num):
#         super(VARNET_pacGAN, self).__init__()
#         self.pac_num = pac_num
#
#         self.Conv1 = nn.Sequential(
#             nn.Conv2d(3*self.pac_dim, 64, kernel_size=5, stride=2),
#             nn.BatchNorm2d(64),
#             nn.LeakyReLU()
#         )
#         self.Conv2 = nn.Sequential(
#             nn.Conv2d(64, 128, kernel_size=5, stride=2),
#             nn.BatchNorm2d(128),
#             nn.LeakyReLU()
#         )
#         self.Conv3 = nn.Sequential(
#             nn.Conv2d(128, 256, kernel_size=5, stride=2),
#             nn.BatchNorm2d(256),
#             nn.LeakyReLU()
#         )
#         self.Conv4 = nn.Sequential(
#             nn.Conv2d(256, 512, kernel_size=5, stride=2),
#             nn.BatchNorm2d(512),
#             nn.LeakyReLU()
#         )
#         self.Linear1 = nn.Sequential(
#             nn.Linear(4* 4 * 512, 1),
#             nn.BatchNorm1d(1),
#             nn.Sigmoid()
#         )
#
#     def forward(self, x):
#
#         print(x.shape)
#         x = self.Conv1(x)
#         print(x.shape)
#
#         x = self.Conv2(x)
#         print(x.shape)
#         x = self.Conv3(x)
#         print(x.shape)
#         x = self.Conv4(x)
#         print(x.shape)
#         x = x.reshape(-1, 4*4*512)
#         x = self.Linear1(x)
#         print(x.shape)
#
#         return x
#

def initialize_models_varnet(img_size, num_condition, pac_dim, pac_var, n_features=100):
    lr = 0.0002
    # if model_kind =='pacGAN':
    #     generator = Generator_pacGAN(n_features)
    #     discriminator = Discriminator_pacGAN(pac_dim)
    #     varnet = VARNET(pac_var)
    # else:
    generator = vanilla_generator(num_condition, img_size*img_size, n_features)
    discriminator = vanilla_discriminator(img_size*img_size, pac_dim)
    varnet = vanilla_varnet(img_size*img_size, pac_var)

    if img_size == 32:
        from MNIST_classifier_32 import Net
        classifier = Net()
        classifier.cuda()
        classifier.load_state_dict(torch.load('/home/sanaz/Ryerson/Projects/VARNETProject/class_dict32.pt'))
    else:
        from MNIST_classifier_64 import Net

        classifier = Net()
        classifier.cuda()
        classifier.load_state_dict(torch.load('/home/sanaz/Ryerson/Projects/VARNETProject/class_dict64.pt'))

    generator.cuda()
    discriminator.cuda()
    varnet.cuda()

    G_optimizer = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    D_optimizer = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
    v_optimizer = optim.Adam(varnet.parameters(), lr=lr, betas=(0.5, 0.999))

    loss = nn.BCELoss()

    return generator, discriminator, varnet, G_optimizer, D_optimizer, v_optimizer, loss, classifier

def initialize_models(img_size, num_condition, pac_dim, n_features=100):
    # if model_kind =='pacGAN':
    #     generator = Generator_pacGAN(n_features)
    #     discriminator = Discriminator_pacGAN(pac_dim)
    # else:
    generator = vanilla_generator(num_condition, img_size*img_size, n_features)
    discriminator = vanilla_discriminator(img_size*img_size, pac_dim)


    if img_size == 32:
        from MNIST_classifier_32 import Net
        classifier = Net()
        classifier.cuda()
        classifier.load_state_dict(torch.load('/home/sanaz/Ryerson/Projects/VARNETProject/class_dict32.pt'))
    else:
        from MNIST_classifier_64 import Net

        classifier = Net()
        classifier.cuda()
        classifier.load_state_dict(torch.load('/home/sanaz/Ryerson/Projects/VARNETProject/class_dict64.pt'))

    generator.cuda()
    discriminator.cuda()

    g_optim = optim.Adam(generator.parameters(), lr=2e-4, betas=(0.5, 0.999))
    d_optim = optim.Adam(discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))

    loss = nn.BCELoss()

    return generator, discriminator, g_optim, d_optim, loss, classifier

# Not the DCGAN from pacGAN paper
def initialize_models_net(model_kind, img_size, n_features=100, pac_dim=4):
    if model_kind =='pacGAN':
        generator = Generator_pacGAN(n_features)
        discriminator = Discriminator_pacGAN(pac_dim)
    else:
        generator = generator_net(n_features)
        discriminator = discriminator_net(pac_dim)
    if img_size == 32:
        from MNIST_classifier_32 import Net
        classifier = Net()
        classifier.cuda()
        classifier.load_state_dict(torch.load('/home/sanaz/Ryerson/Projects/VARNETProject/class_dict32.pt'))
    else:
        from MNIST_classifier_64 import Net

        classifier = Net()
        classifier.cuda()
        classifier.load_state_dict(torch.load('/home/sanaz/Ryerson/Projects/VARNETProject/class_dict64.pt'))

    generator.cuda()
    discriminator.cuda()
    print(summary(generator, (100,100)))
    print(summary(discriminator,(3*pac_dim,64,64)))
    g_optim = optim.Adam(generator.parameters(), lr=2e-4, betas=(0.5, 0.999))
    d_optim = optim.Adam(discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))

    loss = nn.BCELoss()

    return generator, discriminator, g_optim, d_optim, loss, classifier

def initialize_modelsnet_varnet(model_kind, img_size, pac_var, pac_dim, n_features=100):
    lr = 0.0002
    if model_kind =='pacGAN':
        generator = Generator_pacGAN(n_features)
        discriminator = Discriminator_pacGAN(pac_dim)
        varnet = VARNET_pacGAN(pac_var)
    else:
        generator = generator_net(n_features)

        discriminator = discriminator_net(pac_dim)
        varnet = varnet_net(pac_var)

    if img_size == 32:
        from MNIST_classifier_32 import Net
        classifier = Net()
        classifier.cuda()
        classifier.load_state_dict(torch.load('/home/sanaz/Ryerson/Projects/VARNETProject/class_dict32.pt'))
    else:
        from MNIST_classifier_64 import Net

        classifier = Net()
        classifier.cuda()
        classifier.load_state_dict(torch.load('/home/sanaz/Ryerson/Projects/VARNETProject/class_dict64.pt'))

    generator.cuda()
    discriminator.cuda()
    varnet.cuda()
    # print(summary(generator, (100,100)))
    # print(summary(discriminator,(3*pac_dim,64,64)))
    # print(varnet)

    G_optimizer = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    D_optimizer = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
    v_optimizer = optim.Adam(varnet.parameters(), lr=lr, betas=(0.5, 0.999))

    loss = nn.BCELoss()

    return generator, discriminator, varnet, G_optimizer, D_optimizer, v_optimizer, loss, classifier
# initialize_models_net('normal',32,100,1)
# generator, discriminator, g_optim, d_optim, loss, classifier = initialize_models_net('pacGAN',64, 100,1)
# # initialize_modelsnet_varnet('normal', 32, 4, 1, n_features=100)
# generator, discriminator, varnet, G_optimizer, D_optimizer, v_optimizer, loss, classifier = initialize_modelsnet_varnet('pacGAN', 64, 4, 1, n_features=100)
# print(generator.summary(),discriminator.summary())
