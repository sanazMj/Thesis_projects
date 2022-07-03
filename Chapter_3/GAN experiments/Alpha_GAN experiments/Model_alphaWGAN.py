import numpy as np
import torch
import os
from torch import nn
from torch import optim
from torch.nn import functional as F
print_flag= False
#***********************************************
#Encoder and Discriminator has same architecture
#***********************************************
class BasicConv3d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, **kwargs):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm3d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)

class  conv3D_inception_box(nn.Module):
    def __init__(self, in_channels, out_channels, print_flag=False):
        super(conv3D_inception_box, self).__init__()
        self.print_flag = print_flag
        self.conv1 =  BasicConv3d(in_channels, out_channels, kernel_size=1,stride=2)
        self.conv2 = nn.Sequential(
            BasicConv3d(in_channels, out_channels//2, kernel_size=1),
            BasicConv3d(out_channels//2, out_channels, kernel_size=3,stride=2, padding=1))

        self.conv3 = nn.Sequential(
            BasicConv3d(in_channels, out_channels//2, kernel_size=1),
            BasicConv3d(out_channels//2, out_channels, kernel_size=5, stride=2, padding=2) )

        self.conv4 = nn.Sequential(nn.MaxPool3d(kernel_size=3, stride=1, padding=1),
                                   BasicConv3d(in_channels, out_channels, kernel_size=1,stride=2))
        # self.conv1 =  nn.Conv3d(in_channels, out_channels, kernel_size=1)
        # self.conv2 = nn.Sequential(
        #     nn.Conv3d(in_channels, out_channels//2, kernel_size=1),
        #    nn.Conv3d(out_channels//2, out_channels, kernel_size=3, padding=1))
        #
        # self.conv3 = nn.Sequential(
        #     nn.Conv3d(in_channels, out_channels//2, kernel_size=1),
        #     nn.Conv3d(out_channels//2, out_channels, kernel_size=5, padding=2) )
        #
        # self.conv4 = nn.Sequential(nn.MaxPool3d(kernel_size=3, stride=1, padding=1),
        #                            nn.Conv3d(in_channels, out_channels, kernel_size=1))


    def forward(self, x):
        if self.print_flag: print('inception_block', x.shape)
        h1 = self.conv1(x)
        if self.print_flag: print('inception_block', h1.shape)
        h2 = self.conv2(x)
        if self.print_flag: print('inception_block', h2.shape)
        h3 = self.conv3(x)
        if self.print_flag: print('inception_block', h3.shape)
        h4 = self.conv4(x)
        if self.print_flag: print('inception_block', h4.shape)
        output = torch.cat((h1,h2,h3,h4), axis=1)
        if self.print_flag: print('inception_block', output.shape)

        return output


class Discriminator_with_discrete_levels_using_inception(nn.Module):
    def __init__(self, channel=512, out_class=2, target_point=0, is_dis=True):
        super(Discriminator_with_discrete_levels_using_inception, self).__init__()
        self.is_dis = is_dis
        self.channel = channel
        self.target_points = target_point
        n_class = out_class

        # self.l1 = nn.Linear(self.target_points, 1)
        self.l1 = nn.Linear( (channel // 4)*(channel // 8)*(channel // 8)*(channel // 8), 1)

        self.conv1  = conv3D_inception_box(1, channel//8, print_flag=False)
        # self.conv2  = conv3D_inception_box(4 * channel//2, channel//4, print_flag)
        # self.conv3  = conv3D_inception_box(4 * channel//4, channel//8, print_flag)
        self.conv4  = conv3D_inception_box( 4 * channel // 8, n_class , print_flag=False)
        # self.box = nn.Sequential(self.conv1, self.conv2, self.conv3, self.conv4)

        # self.conv22 = nn.Conv3d(4 * channel // 8, channel // 4, kernel_size=4, stride=2, padding=1)
        # self.bn2 = nn.BatchNorm3d(4 * n_class)
        # self.conv3 = nn.Conv3d(channel // 4, channel // 2, kernel_size=4, stride=2, padding=1)
        # self.bn3 = nn.BatchNorm3d(channel // 2)
        # self.conv4 = nn.Conv3d(channel // 2, n_class, kernel_size=4, stride=2, padding=1)
        # self.Leakrelu = nn.LeakyReLU(0.2, inplace=True)
        # self.box = nn.Sequential(self.conv1, self.Leakrelu, self.conv22, self.bn2, self.Leakrelu, self.conv4)

    def forward(self, x, _return_activations=False):
        if print_flag: print('D', x.shape)

        # step = 1 / (self.target_points+1)
        # for i in range(1, self.target_points + 1):
        #     # temp = torch.where(((i*step<=x) & (x< (i+1)*step)),x,0.0)
        #     if i == self.target_points:
        #         temp = torch.where(((i * step <= x)), x, torch.Tensor([0.0]).cuda())
        #     else:
        #         temp = torch.where(((i * step <= x) & (x < (i + 1) * step)), x, torch.Tensor([0.0]).cuda())
        #     if i == 1:
        #         h1 = self.box(temp).reshape(-1, 1)
        #         # print(h1.shape)
        #     else:
        #         h1 = torch.cat((h1, self.box(temp).reshape(h1.shape[0], 1)), axis=1)
        # print(h1.shape)
        # print(self.target_points+1)
        # h1 = F.leaky_relu(self.conv1(x), negative_slope=0.2)
        # if print_flag: print('D', h1.shape)
        #
        # h2 = F.leaky_relu(self.bn2(self.conv2(h1)), negative_slope=0.2)
        # if print_flag: print('D', h2.shape)
        #
        # h3 = F.leaky_relu(self.bn3(self.conv3(h2)), negative_slope=0.2)
        # if print_flag: print('D', h3.shape)
        #
        # # h4 = F.leaky_relu(self.bn4(self.conv4(h3)), negative_slope=0.2)
        # h4 = self.conv4(h3)
        # if print_flag: print('D', h4.shape)

        # h5 = self.conv5(h4)
        # if print_flag: print('D',h5.shape)
        # print(h1.shape)

        # h1 = h1.reshape(h1.shape[0],h1.shape[1])
        # h1 = self.box(x)
        h1 = self.conv1(x)
        if print_flag: print('D', h1.shape)

        # h1 = self.Leakrelu(h1)
        # if print_flag: print('D', h1.shape)
        #
        # h1 = self.conv22(h1)
        # if print_flag: print('D', h1.shape)




        h1 = self.conv4(h1)
        if print_flag: print('D', h1.shape)

        # h1 = self.bn2(h1)
        # if print_flag: print('D', h1.shape)
        # h1 = self.Leakrelu(h1)
        # if print_flag: print('D', h1.shape)

        h1 = h1.reshape(h1.shape[0],-1)
        if print_flag: print('D', h1.shape)

        output = self.l1(h1)
        if print_flag: print('D', output.shape)

        # output = self.avg1(h1)
        # output = output.reshape(output.shape[0],output.shape[2])
        #
        # # print(output.shape)
        # output = self.sigmoid(output)
        # output = h5
        # output = h4
        # print(output)
        return output


class Discriminator_with_discrete_levels(nn.Module):
    def __init__(self, channel=512, out_class=1,target_point=0, is_dis=True):
        super(Discriminator_with_discrete_levels, self).__init__()
        self.is_dis = is_dis
        self.channel = channel
        self.target_points = target_point
        n_class = out_class

        self.conv1 = nn.Conv3d(1, channel // 8, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv3d(channel // 8, channel // 4, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm3d(channel // 4)
        self.conv3 = nn.Conv3d(channel // 4, channel // 2, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm3d(channel // 2)
        self.conv4 = nn.Conv3d(channel // 2, n_class, kernel_size=4, stride=2, padding=1)
        self.Leakrelu  = nn.LeakyReLU(0.2, inplace=True)
        self.box = nn.Sequential(self.conv1, self.Leakrelu, self.conv2, self.bn2, self.Leakrelu,
                                 self.conv3, self.bn3, self.Leakrelu, self.conv4
                                 )
        self.l1 = nn.Linear(self.target_points, 1)
        self.sigmoid = nn.Sigmoid()
        self.avg1 = nn.AvgPool1d(self.target_points)
        # self.conv4 = nn.Conv3d(channel//2, channel, kernel_size=4, stride=2, padding=1)
        # self.bn4 = nn.BatchNorm3d(channel)
        # self.conv5 = nn.Conv3d(channel, n_class, kernel_size=4, stride=1, padding=0)

    def forward(self, x, _return_activations=False):
        # if print_flag: print('D', x.shape)
        # h1 = self.box(x)
        # h1 = h1.reshape(h1.shape[0],1)
        # print(h1.shape)
        step = 1/(self.target_points+1)
        for i in range(1,self.target_points+1):
            # temp = torch.where(((i*step<=x) & (x< (i+1)*step)),x,0.0)
            if i == self.target_points:
                temp = torch.where(((i * step <= x)), x, torch.Tensor([0.0]).cuda())
            else:
                temp = torch.where(((i * step <= x) & (x < (i + 1) * step)), x, torch.Tensor([0.0]).cuda())
            # print(temp.shape)
            if i==1:
                h1 = self.box(temp).reshape(-1,1)
            else:
                h1 = torch.cat((h1, self.box(temp).reshape(h1.shape[0],1)),axis=1)
        # print(h1.shape)
        # print(self.target_points+1)
        # h1 = F.leaky_relu(self.conv1(x), negative_slope=0.2)
        # if print_flag: print('D', h1.shape)
        #
        # h2 = F.leaky_relu(self.bn2(self.conv2(h1)), negative_slope=0.2)
        # if print_flag: print('D', h2.shape)
        #
        # h3 = F.leaky_relu(self.bn3(self.conv3(h2)), negative_slope=0.2)
        # if print_flag: print('D', h3.shape)
        #
        # # h4 = F.leaky_relu(self.bn4(self.conv4(h3)), negative_slope=0.2)
        # h4 = self.conv4(h3)
        # if print_flag: print('D', h4.shape)

        # h5 = self.conv5(h4)
        # if print_flag: print('D',h5.shape)
        # print(h1.shape)
        # output = self.l1(h1)
        output = self.avg1(h1)

        # output = self.sigmoid(output)
        # output = h5
        # output = h4
        # print(output)
        return output

class Discriminator(nn.Module):
    def __init__(self, channel=512,out_class=1,is_dis =True):
        super(Discriminator, self).__init__()
        self.is_dis=is_dis
        self.channel = channel
        n_class = out_class 
        
        self.conv1 = nn.Conv3d(1, channel//8, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv3d(channel//8, channel//4, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm3d(channel//4)
        self.conv3 = nn.Conv3d(channel//4, channel//2, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm3d(channel//2)
        self.conv4 = nn.Conv3d(channel//2, n_class, kernel_size=4, stride=2, padding=1)

        # self.conv4 = nn.Conv3d(channel//2, channel, kernel_size=4, stride=2, padding=1)
        # self.bn4 = nn.BatchNorm3d(channel)
        # self.conv5 = nn.Conv3d(channel, n_class, kernel_size=4, stride=1, padding=0)
        
    def forward(self, x, _return_activations=False):
        if print_flag: print('D', x.shape)
        h1 = F.leaky_relu(self.conv1(x), negative_slope=0.2)
        if print_flag: print('D',h1.shape)

        h2 = F.leaky_relu(self.bn2(self.conv2(h1)), negative_slope=0.2)
        if print_flag: print('D',h2.shape)

        h3 = F.leaky_relu(self.bn3(self.conv3(h2)), negative_slope=0.2)
        if print_flag: print('D',h3.shape)

        # h4 = F.leaky_relu(self.bn4(self.conv4(h3)), negative_slope=0.2)
        h4  = self.conv4(h3)
        if print_flag: print('D',h4.shape)

        # h5 = self.conv5(h4)
        # if print_flag: print('D',h5.shape)

        # output = h5
        output = h4
        return output
    
class Code_Discriminator(nn.Module):
    def __init__(self, code_size=100,num_units=750):
        super(Code_Discriminator, self).__init__()
        n_class = 1
        self.l1 = nn.Sequential(nn.Linear(code_size, num_units),
                                nn.BatchNorm1d(num_units),
                                nn.LeakyReLU(0.2,inplace=True))
        self.l2 = nn.Sequential(nn.Linear(num_units, num_units),
                                nn.BatchNorm1d(num_units),
                                nn.LeakyReLU(0.2,inplace=True))
        self.l3 = nn.Linear(num_units, 1)
        
    def forward(self, x):
        if print_flag: print('CD',x.shape)

        h1 = self.l1(x)
        if print_flag: print('CD',h1.shape)

        h2 = self.l2(h1)
        if print_flag: print('CD',h2.shape)

        h3 = self.l3(h2)
        if print_flag: print('CD',h3.shape)

        output = h3
            
        return output

class Generator(nn.Module):
    def __init__(self, noise:int=100, channel:int=64):
        super(Generator, self).__init__()
        _c = channel

        self.relu = nn.ReLU()
        self.noise = noise
        self.tp_conv1 = nn.ConvTranspose3d(noise, _c*8, kernel_size=4, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm3d(_c*8)
        
        self.tp_conv2 = nn.Conv3d(_c*8, _c*4, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(_c*4)
        
        self.tp_conv3 = nn.Conv3d(_c*4, 1, kernel_size=3, stride=1, padding=1, bias=False)
        # self.bn3 = nn.BatchNorm3d(_c*2)
        
        # self.tp_conv4 = nn.Conv3d(_c*2, _c, kernel_size=3, stride=1, padding=1, bias=False)
        # self.bn4 = nn.BatchNorm3d(_c)
        #
        # self.tp_conv5 = nn.Conv3d(_c, 1, kernel_size=3, stride=1, padding=1, bias=False)
        
    def forward(self, noise):
        if print_flag: print('G', noise.shape)

        noise = noise.view(-1,self.noise,1,1,1)
        if print_flag: print('G', noise.shape)

        h = self.tp_conv1(noise)
        if print_flag: print('G', h.shape)

        h = self.relu(self.bn1(h))
        if print_flag: print('G', h.shape)

        h = F.interpolate(h,scale_factor = 2)
        if print_flag: print('G', h.shape)

        h = self.tp_conv2(h)
        if print_flag: print('G', h.shape)

        h = self.relu(self.bn2(h))
        if print_flag: print('G', h.shape)

        h = F.interpolate(h,scale_factor = 2)
        if print_flag: print('G', h.shape)

        h = self.tp_conv3(h)
        if print_flag: print('G', h.shape)
        #
        # h = self.relu(self.bn3(h))
        # if print_flag: print('G', h.shape)
        #
        # h = F.upsample(h,scale_factor = 2)
        # if print_flag: print('G', h.shape)
        # h = self.tp_conv4(h)
        # if print_flag: print('G', h.shape)
        # h = self.relu(self.bn4(h))
        # if print_flag: print('G', h.shape)
        #
        # h = F.upsample(h,scale_factor = 2)
        # if print_flag: print('G', h.shape)
        # h = self.tp_conv5(h)
        # if print_flag: print('G', h.shape)

        # ste = nn.Tanh()
        ste = torch.nn.Sigmoid()
        output = ste(h)
        if print_flag: print('G', output.shape)

        return output
