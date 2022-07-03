# https://github.com/black0017/3D-GAN-pytorch
# https://github.com/black0017/3D-GAN-pytorch/blob/master/notebooks/3D_GAN_pytorch.ipynb
import torch
import torch.nn as nn
# from torchsummary import summary

"""
Implementation based on original paper NeurIPS 2016
https://papers.nips.cc/paper/6096-learning-a-probabilistic-latent-space-of-object-shapes-via-3d-generative-adversarial-modeling.pdf
"""

print_flag = False
class Discriminator(torch.nn.Module):
    def __init__(self, in_channels=3, dim=64, out_conv_channels=512, model='GAN'):
        super(Discriminator, self).__init__()
        conv1_channels = int(out_conv_channels / 8)
        conv2_channels = int(out_conv_channels / 4)
        conv3_channels = int(out_conv_channels / 2)
        self.out_conv_channels = out_conv_channels
        self.conv_count = 4
        self.out_dim1 = int(dim[0] / (2**self.conv_count))
        self.out_dim2 = int(dim[1] / (2**self.conv_count))
        self.out_dim3 = int(dim[2] / (2**self.conv_count))
        self.model = model
        self.conv1 = nn.Sequential(
            nn.Conv3d(
                in_channels=in_channels, out_channels=conv1_channels, kernel_size=4,
                stride=2, padding=1, bias=False
            ),
            nn.BatchNorm3d(conv1_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(
                in_channels=conv1_channels, out_channels=conv2_channels, kernel_size=4,
                stride=2, padding=1, bias=False
            ),
            nn.BatchNorm3d(conv2_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv3d(
                in_channels=conv2_channels, out_channels=conv3_channels, kernel_size=4,
                stride=2, padding=1, bias=False
            ),
            nn.BatchNorm3d(conv3_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # self.conv3 = nn.Sequential(
        #     nn.Conv3d(
        #         in_channels=conv2_channels, out_channels=out_conv_channels, kernel_size=4,
        #         stride=2, padding=1, bias=False
        #     ),
        #     nn.BatchNorm3d(out_conv_channels),
        #     nn.LeakyReLU(0.2, inplace=True)
        # )
        self.conv4 = nn.Sequential(
            nn.Conv3d(
                in_channels=conv3_channels, out_channels=out_conv_channels, kernel_size=4,
                stride=2, padding=1, bias=False
            ),
            nn.BatchNorm3d(out_conv_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.out = nn.Sequential(
            nn.Linear(out_conv_channels * self.out_dim1 * self.out_dim2 * self.out_dim3, 1),
            nn.Sigmoid(),
        )
        self.out1 = nn.Sequential(
            nn.Linear(out_conv_channels * self.out_dim1 * self.out_dim2 * self.out_dim3, 1)
        )

    def forward(self, x):
        if print_flag == True : print('Discriminator', x.shape)
        x = self.conv1(x)
        if print_flag == True : print('Discriminator',x.shape)
        x = self.conv2(x)
        if print_flag == True : print('Discriminator',x.shape)
        x = self.conv3(x)
        if print_flag == True : print('Discriminator',x.shape)
        x = self.conv4(x)
        if print_flag == True : print('Discriminator',x.shape)

        # Flatten and apply linear + sigmoid
        x = x.view(-1, self.out_conv_channels * self.out_dim1 * self.out_dim2 * self.out_dim3)
        if print_flag == True : print('Discriminator',x.shape)
        if self.model == 'GAN':

            x = self.out(x)
        else:
            x = self.out1(x)

        if print_flag == True : print('Discriminator',x.shape)

        return x


class Generator(torch.nn.Module):
    def __init__(self, in_channels=512, out_dim=[64,64,64], out_channels=1, noise_dim=200, activation="Tanh"):
        super(Generator, self).__init__()
        self.in_channels = in_channels
        self.noise_dim = noise_dim
        self.out_dim = out_dim
        self.num_conv= 4
        self.in_dim1 = int(self.out_dim[0] / (2**self.num_conv))
        self.in_dim2 = int(self.out_dim[1] / (2**self.num_conv))
        self.in_dim3 = int(self.out_dim[2] / (2**self.num_conv))
        conv1_out_channels = int(self.in_channels / 2)
        conv2_out_channels = int(conv1_out_channels / 2)
        conv3_out_channels = int(conv2_out_channels / 2)
        # print('g',self.in_channels, self.noise_dim, self.out_dim, self.in_dim)
        self.linear = torch.nn.Linear(self.noise_dim, self.in_channels * self.in_dim1 * self.in_dim2 * self.in_dim3)

        self.conv1 = nn.Sequential(
            nn.ConvTranspose3d(
                in_channels=in_channels, out_channels=conv1_out_channels, kernel_size=(4, 4, 4),
                stride=2, padding=1, bias=False
            ),
            nn.BatchNorm3d(conv1_out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.ConvTranspose3d(
                in_channels=conv1_out_channels, out_channels=conv2_out_channels, kernel_size=(4, 4, 4),
                stride=2, padding=1, bias=False
            ),
            nn.BatchNorm3d(conv2_out_channels),
            nn.ReLU(inplace=True)
        )
        # self.conv3 = nn.Sequential(
        #     nn.ConvTranspose3d(
        #         in_channels=conv2_out_channels, out_channels=out_channels, kernel_size=(4, 4, 4),
        #         stride=2, padding=1, bias=False
        #     ),
        #     nn.BatchNorm3d(out_channels),
        #     nn.ReLU(inplace=True)
        # )
        self.conv3 = nn.Sequential(
            nn.ConvTranspose3d(
                in_channels=conv2_out_channels, out_channels=conv3_out_channels, kernel_size=(4, 4, 4),
                stride=2, padding=1, bias=False
            ),
            nn.BatchNorm3d(conv3_out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.ConvTranspose3d(
                in_channels=conv3_out_channels, out_channels=out_channels, kernel_size=(4, 4, 4),
                stride=2, padding=1, bias=False
            )
        )
        if activation == "sigmoid":
            self.out = torch.nn.Sigmoid()
        elif activation == "Tanh":
            self.out = torch.nn.Tanh()
        else:
            self.out = torch.nn.ReLU()
    def project(self, x):
        """
        projects and reshapes latent vector to starting volume
        :param x: latent vector
        :return: starting volume
        """
        return x.view(-1, self.in_channels, self.in_dim1, self.in_dim2, self.in_dim3)

    def forward(self, x):
        if print_flag == True : print('Generator',x.shape)
        x = self.linear(x)
        if print_flag == True : print('Generator',x.shape)
        x = self.project(x)
        if print_flag == True : print('Generator',x.shape)
        x = self.conv1(x)
        if print_flag == True : print('Generator',x.shape)
        x = self.conv2(x)
        if print_flag == True : print('Generator',x.shape)
        x = self.conv3(x)
        if print_flag == True : print('Generator',x.shape)
        x = self.conv4(x)
        if print_flag == True : print('Generator',x.shape)
        return self.out(x)


def test_gan3d():
    noise_dim = 50
    in_channels = 16
    dim = [16,16,16]   # cube volume
    model_generator = Generator(in_channels=in_channels, out_dim=dim, out_channels=1, noise_dim=noise_dim)
    noise = torch.rand(10, noise_dim)
    generated_volume = model_generator(noise)
    print("Generator output shape", generated_volume.shape)
    model_discriminator = Discriminator(in_channels=1, dim=dim, out_conv_channels=in_channels)
    out = model_discriminator(generated_volume)
    print("Discriminator output", out)
    summary(model_generator, (10, noise_dim))
    summary(model_discriminator, (10, 1,16, 16, 16))


# test_gan3d()
#
# # files_epoch =  [799]
# # # files_epoch =[999]
# # files_epoch = list(range(100,800, step_report)) #+ [799]
# dir = '/home/sanaz/Ryerson/Projects/tumorGAN/GAN_simple_3D/results/'
# from torch import nn, optim
# from torch.utils.data import DataLoader
# from torchvision import datasets
# from torch.autograd import Variable
# from tumor_to_isocenter_mapping.Sphere_packing import check_files_convexity
# from GAN_3D.helper import GAN_3D_iso_sphere_full_evaluation
# generator_file = 'generator_epoch_300id_150.pt'
# # generator = Generator(in_channels=16, out_dim=[16,16,16], out_channels=1, noise_dim=100)
# # generator = generator.load_state_dict(torch.load(dir + generator_file))
# generator = torch.load(dir + generator_file)
# generator.eval()
# import numpy as np
# cuda = True if torch.cuda.is_available() else False
# Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
# generator.cuda()
# count_z = 40000
# z = Variable(Tensor(np.random.normal(0, 1, (count_z, 100))))
# fake_imgs = generator(z)
# dict1 = GAN_3D_iso_sphere_full_evaluation(fake_imgs, 1, False, [16,16,16])
# check_files_convexity(file='', id_name='150id_epoch_300',obj=dict1, save_omega=False)