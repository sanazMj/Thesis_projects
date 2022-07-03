
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd.variable import Variable
import torch as T
from Synthetic_data.utils import MinibatchDiscrimination
####################
###### New VARNET model with multiple packing factors
#################################
class VARNET_Vpacks(nn.Module):
    def __init__(self, n_in, pac_var0, pac_var1, pac_var2):
        super(VARNET_Vpacks, self).__init__()
        self.n_in0 = n_in * pac_var0
        self.n_in1 = n_in * pac_var1
        self.n_in2 = n_in * pac_var2

        self.n_out = 1
        self.fc01 = nn.Sequential(
                    nn.Linear(self.n_in0, 1024),
                    nn.LeakyReLU(0.2),
                    nn.Dropout(0.3)
                    )
        self.fc02 = nn.Sequential(
                    nn.Linear(self.n_in1, 1024),
                    nn.LeakyReLU(0.2),
                    nn.Dropout(0.3)
                    )
        self.fc03 = nn.Sequential(
                    nn.Linear(self.n_in2, 1024),
                    nn.LeakyReLU(0.2),
                    nn.Dropout(0.3)
                    )

        # self.fc0 = nn.Sequential(
        #             nn.Linear(1024 , 1024),
        #             nn.LeakyReLU(0.2),
        #             nn.Dropout(0.3)
        #             )
        self.fc1 = nn.Sequential(
                    nn.Linear(1024, 512),
                    nn.LeakyReLU(0.2),
                    nn.Dropout(0.3)
                    )
        self.fc2 = nn.Sequential(
                    nn.Linear(512, 256),
                    nn.LeakyReLU(0.2),
                    nn.Dropout(0.3)
                    )
        self.fc3 = nn.Sequential(
                    nn.Linear(256, self.n_out),
                    nn.Sigmoid()
                    )
    def forward(self, x0, x1, x2):
        # print('v-last', x.shape)
        # print('v',x0.shape,x1.shape, x2.shape)

        x0 = x0.view(-1, self.n_in0)
        x1 = x1.view(-1, self.n_in1)
        x2 = x2.view(-1, self.n_in2)
        # print('v',x0.shape,x1.shape, x2.shape)
        x0 = self.fc01(x0)
        x1 = self.fc02(x1)
        x2 = self.fc03(x2)
        # print('v',x0.shape,x1.shape, x2.shape)

        x = torch.cat((x0,x1,x2),axis=0)
        # print(x.shape)
        # print('v-last', x.shape
        # x = self.fc0(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        # print('v-last', x.shape)
        return x



class VARNET(nn.Module):
    def __init__(self, n_in, pac_var):
        super(VARNET, self).__init__()
        self.n_in = n_in * pac_var
        self.n_out = 1
        self.fc0 = nn.Sequential(
                    nn.Linear(self.n_in, 1024),
                    nn.LeakyReLU(0.2),
                    nn.Dropout(0.3)
                    )
        self.fc1 = nn.Sequential(
                    nn.Linear(1024, 512),
                    nn.LeakyReLU(0.2),
                    nn.Dropout(0.3)
                    )
        self.fc2 = nn.Sequential(
                    nn.Linear(512, 256),
                    nn.LeakyReLU(0.2),
                    nn.Dropout(0.3)
                    )
        self.fc3 = nn.Sequential(
                    nn.Linear(256, self.n_out),
                    nn.Sigmoid()
                    )
    def forward(self, x):
        # print('v-last', x.shape)
        x = x.view(-1, self.n_in)
        # print('v-last', x.shape)
        x = self.fc0(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        # print('v-last', x.shape)
        return x


# Generator from PacGAN paper (Synthetic data, architectures)

class Generator_pacGAN(nn.Module):
    # # Generator from PacGAN paper (Synthetic data, architectures)
        def __init__(self, n_features, n_out):
            super(Generator_pacGAN, self).__init__()
            self.n_features = n_features
            self.n_out = n_out
            self.input1 = nn.Sequential(
                nn.Linear(self.n_features, 400),
                nn.ReLU()
            )
            self.hidden1 = nn.Sequential(
                nn.Linear(400, 400),
                nn.BatchNorm1d(400),
                nn.ReLU()
            )
            self.hidden2 = nn.Sequential(
                nn.Linear(400, 400),
                nn.ReLU()
            )
            self.hidden3 = nn.Sequential(
                nn.Linear(400, 400),
                nn.ReLU()
            )
            self.hidden4 = nn.Sequential(
                nn.Linear(400, 400),
                nn.ReLU()
            )
            self.fc3 = nn.Sequential(
                nn.Linear(400, self.n_out),
                nn.Identity()
            )

        def forward(self, x):
            # print('G', x.shape)
            x = self.input1(x)
            # print('G', x.shape)

            x = self.hidden1(x)
            # print('G', x.shape)

            x = self.hidden2(x)
            # print('G', x.shape)

            x = self.hidden3(x)
            # print('G', x.shape)

            x = self.hidden4(x)
            # print('G', x.shape)

            x = self.fc3(x)
            # print('G', x.shape)

            # x = x.view(-1, 1, 28, 28)
            return x


# %%

# Discriminator from PacGAN paper (Synthetic data, architectures)

class Discriminator_pacGAN(nn.Module):
        def __init__(self, n_in, pac_dim, minibatch_net):
            super(Discriminator_pacGAN, self).__init__()
            self.n_in = n_in * pac_dim
            self.n_out = 1
            self.n_channels = 5
            self.minibatch_net = minibatch_net
            self.weight = T.nn.Parameter(T.Tensor(5 * 200, 2))
            self.weight1 = T.nn.Parameter(T.Tensor(5 * 200, 200))

            # self.input1 = Sequential(
            #             nn.functional.linear(input, self.weight, self.bias),
            #             nn.functional.max_pool1d(a.unsqueeze(-3), kernel_size=self.n_channels)
            # )
            # self.hidden1 = maxout2
            # self.hidden2 = Sequential(
            #             nn.functional.linear(input, self.weight, self.bias),
            #             nn.functional.max_pool1d(a.unsqueeze(-3), kernel_size=self.n_channels)
            # )
            if self.minibatch_net:
                self.fc3 = nn.Sequential(
                    nn.Linear(200*2, self.n_out),
                    nn.Sigmoid()
                )
            else:
                self.fc3 = nn.Sequential(
                    nn.Linear(200, self.n_out),
                    nn.Sigmoid()
                )

        def forward(self, x):
            print(x)
            x = nn.functional.linear(x, self.weight)
            print(x)

            # print('d', x.shape)
            x = nn.functional.max_pool1d(x.unsqueeze(-3), kernel_size=self.n_channels)
            print(x)

            # print('d', x.shape)
            x = x.squeeze()
            # print('d', x.shape)
            print(x)

            x = nn.functional.linear(x, self.weight1)
            print(x)

            # print('d', x.shape)
            x = nn.functional.max_pool1d(x.unsqueeze(-3), kernel_size=self.n_channels)
            # print('d', x.shape)
            x = x.squeeze()
            # print('d', x.shape)

            x = nn.functional.linear(x, self.weight1)
            # print('d', x.shape)
            x = nn.functional.max_pool1d(x.unsqueeze(-3), kernel_size=self.n_channels)
            # print('d', x.shape)
            x = x.squeeze()
            # print('d', x.shape)
            if self.minibatch_net:
                x = self.minibatch_net(x)
            # print('d', x.shape)
            x = self.fc3(x)
            # print('d', x.shape)
            # x = x.reshape(-1, 1)
            # print('d', x.shape)

            # x = x.view(-1, 1, 28, 28)
            return x


# %%
class Generator(nn.Module):
    def __init__(self, n_features, n_out):
        super(Generator, self).__init__()
        self.n_features = n_features
        self.n_out = n_out
        self.fc0 = nn.Sequential(
                    nn.Linear(self.n_features, 256),
                    nn.LeakyReLU(0.2)
                    )
        self.fc1 = nn.Sequential(
                    nn.Linear(256, 512),
                    nn.LeakyReLU(0.2)
                    )
        self.fc2 = nn.Sequential(
                    nn.Linear(512, 1024),
                    nn.LeakyReLU(0.2)
                    )
        self.fc3 = nn.Sequential(
                    nn.Linear(1024, self.n_out),
                    nn.Tanh()
                    )
    def forward(self, x):
        x = self.fc0(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        # x = x.view(-1, 1, 28, 28)
        return x




class Discriminator(nn.Module):
    def __init__(self, n_in, pac_dim,minibatch_net):
        super(Discriminator, self).__init__()
        self.n_in = n_in * pac_dim
        self.n_out = 1
        self.minibatch_net = minibatch_net
        self.fc0 = nn.Sequential(
                    nn.Linear(self.n_in, 1024),
                    nn.LeakyReLU(0.2),
                    nn.Dropout(0.3)
                    )
        self.fc1 = nn.Sequential(
                    nn.Linear(1024, 512),
                    nn.LeakyReLU(0.2),
                    nn.Dropout(0.3)
                    )
        self.fc2 = nn.Sequential(
                    nn.Linear(512, 256),
                    nn.LeakyReLU(0.2),
                    nn.Dropout(0.3)
                    )
        if self.minibatch_net:
            self.fc3 = nn.Sequential(
                        nn.Linear(256*2, self.n_out),
                        nn.Sigmoid()
                        )
        else:
            self.fc3 = nn.Sequential(
                        nn.Linear(256, self.n_out),
                        nn.Sigmoid()
                        )
    def forward(self, x):
        x = x.view(-1, self.n_in)
        x = self.fc0(x)
        x = self.fc1(x)
        x = self.fc2(x)
        # print('d', x.shape)
        if self.minibatch_net:
            x = self.minibatch_net(x)
        # print('d', x.shape)

        x = self.fc3(x)
        # print('d', x.shape)

        return x

# %%

def initialize_models_varnet(model_kind, minibatch_net, pac_var, same_creation_type, pac_dim, n_features=128, n_in=2, n_out=2):

    if model_kind =='pacGAN':
        generator = Generator_pacGAN(n_features, n_out)
        discriminator = Discriminator_pacGAN(n_in, pac_dim, minibatch_net)
        if same_creation_type== 0 or same_creation_type == 1:
            varnet = VARNET(n_in, pac_var)
        else:
            varnet = VARNET_Vpacks(n_in, pac_var[0], pac_var[1], pac_var[2])
    else:
        generator = Generator(n_features, n_out)
        discriminator = Discriminator(n_in, pac_dim, minibatch_net)
        if same_creation_type== 0 or same_creation_type == 1:
            varnet = VARNET(n_in, pac_var)
        else:
            varnet = VARNET_Vpacks(n_in, pac_var[0], pac_var[1], pac_var[2])


    generator.cuda()
    discriminator.cuda()
    varnet.cuda()

    g_optim = optim.Adam(generator.parameters(), lr=2e-4)
    d_optim = optim.Adam(discriminator.parameters(), lr=2e-4)
    v_optim = optim.Adam(varnet.parameters(), lr=2e-4)

    loss = nn.BCELoss()

    return generator, discriminator, varnet, g_optim, d_optim, v_optim, loss

def initialize_models(model_kind, minibatch_net, n_features=128, n_in=2, n_out=2, pac_dim=4):


    if model_kind =='pacGAN':
        if minibatch_net:
            minibatch_net = MinibatchDiscrimination(200, 200, 5, Minibatch_kind='L1 Norm')
            minibatch_net.cuda()
        generator = Generator_pacGAN(n_features, n_out)
        discriminator = Discriminator_pacGAN(n_in, pac_dim, minibatch_net)
    else:
        if minibatch_net:
            minibatch_net = MinibatchDiscrimination(256, 256, 5, Minibatch_kind='L1 Norm')
            minibatch_net.cuda()
        generator = Generator(n_features, n_out)
        discriminator = Discriminator(n_in, pac_dim, minibatch_net)

    generator.cuda()
    discriminator.cuda()

    g_optim = optim.Adam(generator.parameters(), lr=2e-4)
    d_optim = optim.Adam(discriminator.parameters(), lr=2e-4)

    loss = nn.BCELoss()

    return generator, discriminator, g_optim, d_optim, loss