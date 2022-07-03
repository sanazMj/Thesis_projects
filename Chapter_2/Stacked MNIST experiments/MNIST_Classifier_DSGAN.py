import os, time
import matplotlib.pyplot as plt
import itertools
import pickle
import imageio
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

torch.cuda.is_available()

#
# # %%
#
# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#
#         self.Conv1 = nn.Sequential(
#             nn.Conv2d(1, 64, kernel_size=5, padding=2, stride=2),
#             nn.LeakyReLU()
#         )
#         self.Conv2 = nn.Sequential(
#             nn.Conv2d(64, 128, kernel_size=5, padding=2, stride=2),
#             nn.BatchNorm2d(128),
#             nn.LeakyReLU()
#         )
#         self.Conv3 = nn.Sequential(
#             nn.Conv2d(128, 256, kernel_size=5, padding=2, stride=2),
#             nn.BatchNorm2d(256),
#             nn.LeakyReLU()
#         )
#         self.Conv4 = nn.Sequential(
#             nn.Conv2d(256, 512, kernel_size=5, padding=2, stride=2),
#             nn.LeakyReLU()
#         )
#         self.Linear1 = nn.Sequential(
#             nn.Linear(2 * 2 * 512, 1),
#             nn.BatchNorm1d(1)
#         )
#
#     def forward(self, x):
#         # print(x.shape)
#         x = self.Conv1(x)
#         # print(x.shape)
#         x = self.Conv2(x)
#         # print(x./shape)
#
#         x = self.Conv3(x)
#         # print(x.shape)
#         x = self.Conv4(x)
#         # print(x.shape)
#         x = x.reshape(-1, 512 * 2 * 2)
#         # print(x.shape)
#         x = self.Linear1(x)
#         # print(x.shape)
#         x = F.log_softmax(x, dim=1)
#         return x
#
#     # %%


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(64 * 12 * 12, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)

        return output


# %%

def binarize_images(image, img_size=28, threshold=0.5):
    image = image > threshold
    image = image.float()

    return image


# %%

# data_loader
# training parameters
batch_size = 128
lr = 0.0002
train_epoch = 100
img_size = 28
transform = transforms.Compose([
    transforms.Scale(img_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,), std=(0.5,))
])
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=True, download=True, transform=transform),
    batch_size=batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=False, download=True, transform=transform),
    batch_size=batch_size, shuffle=True)

# # Binary MNIST Train data
# index = 0
# binarized_train_data = []
# for  x_, y_ in (train_loader):

#         if index == 0:
#           binarized_input = binarize_images(x_)
#           label = y_
#           index+=1
#         else:
#           label = torch.cat([label,y_],0)
#           binarized_input = torch.cat([binarized_input, binarize_images(x_)],0)

# binarized_train_data =torch.utils.data.TensorDataset( binarized_input, label)

# train_loader_binarized =  torch.utils.data.DataLoader(binarized_train_data, batch_size =batch_size ,shuffle=True)

# # Binary MNIST Test data
# index = 0
# binarized_test_data = []
# for  x_, y_ in (test_loader):

#         if index == 0:
#           binarized_input = binarize_images(x_)
#           label = y_
#           index+=1
#         else:
#           label = torch.cat([label,y_],0)
#           binarized_input = torch.cat([binarized_input, binarize_images(x_)],0)

# binarized_test_data =torch.utils.data.TensorDataset( binarized_input, label)

# test_loader_binarized =  torch.utils.data.DataLoader(binarized_test_data, batch_size =batch_size ,shuffle=True)

# %%

classifier = Net()
classifier.cuda()

# negative log likelihood loss loss
loss_NLL = nn.NLLLoss()

# Adam optimizer
classifier_optimizer = optim.Adam(classifier.parameters(), lr=lr, betas=(0.5, 0.999))

# %%

num_epochs = 10
for epoch in range(num_epochs):  # loop over the dataset multiple times
    print(epoch)
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        # print(inputs.shape, labels.shape)
        inputs = inputs.cuda()
        labels = labels.cuda()
        # zero the parameter gradients
        classifier_optimizer.zero_grad()

        # forward + backward + optimize
        outputs = classifier(inputs)
        loss = loss_NLL(outputs, labels)
        # print(loss)
        loss.backward()
        classifier_optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:  # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')

# %%

torch.save(classifier.state_dict(), '/home/sanaz/Ryerson/Projects/GAN_Main_Project/VARNET_MNIST/class_dict.pt')

# %%

correct = 0
total = 0
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        images = images.cuda()
        labels = labels.cuda()
        outputs = classifier(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))



# %%


