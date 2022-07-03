import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd.variable import Variable
import torch.nn.functional as F
import torch as T
from torchsummary import summary

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(57600, 128)

        # self.fc1 = nn.Linear(64 * 12 * 12, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # print(x.shape)
        x = self.conv1(x)
        # print(x.shape)

        x = F.relu(x)
        # print(x.shape)
        x = self.conv2(x)
        # print(x.shape)
        x = F.relu(x)
        # print(x.shape)

        x = F.max_pool2d(x, 2)
        # print(x.shape)

        x = self.dropout1(x)
        # print(x.shape)

        x = torch.flatten(x, 1)
        # print(x.shape)

        x = self.fc1(x)
        # print(x.shape)/

        x = F.relu(x)
        # print(x.shape)

        x = self.dropout2(x)
        # print(x.shape)

        x = self.fc2(x)
        # print(x.shape)

        output = F.log_softmax(x, dim=1)
        # print(x.shape)

        return output



def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()
