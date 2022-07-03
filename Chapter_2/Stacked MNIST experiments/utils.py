

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd.variable import Variable
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
import torch.distributions as D
from torch.nn import functional as F
import torch.utils.data

import random
from torchvision.models.inception import inception_v3
from numpy import ones
from numpy import expand_dims
from numpy import log
from numpy import mean
from numpy import std
from numpy import exp

import matplotlib.pyplot as plt
import imageio
import numpy as np
from scipy.stats import entropy

def create_prob(labels):
    modes = {}
    labels = labels.type(torch.LongTensor).cpu().numpy()
    for i in labels:
        i = np.array(i)
        if tuple(i) in list(modes.keys()):
            modes[tuple(i)] += 1
        else:
            modes[tuple(i)] = 1

    num_mode = len(modes.keys())

    p = list(modes.values())
    p = p / np.sum(p)
    return p
def load_data(batch_size, img_size):
    dataset_path = '/home/sanaz/Ryerson/Projects/VARNETProject/Stacked_MNIST/data/'

    Stacked_data = torch.load(dataset_path + '/Stacked_data_'+str(img_size)+'_dataset.pt')
    train_loader_Stacked = torch.utils.data.DataLoader(Stacked_data, batch_size=batch_size, shuffle=True)
    label_list = torch.Tensor(np.load(dataset_path + '/Labels_'+ str(img_size)+'.npy'))
    prob = create_prob(label_list)
    return train_loader_Stacked, prob

def load_data_create(num_training_sample, img_size):
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,), std=(0.5,))
    ])
    # Create the dataloader
    dataloader = torch.utils.data.DataLoader(
        MNIST('data', train=True, download=True, transform=transform),
        batch_size=100, shuffle=True)

    index = 0
    data = []
    for x_, y_ in (dataloader):
        if index == 0:
            input_data = x_
            label = y_
            index += 1
        else:
            label = torch.cat([label, y_], 0)
            input_data = torch.cat([input_data, x_], 0)


    ids = torch.randint(0, input_data.shape[0], size=(num_training_sample, 3))
    X_training = torch.zeros(ids.shape[0], ids.shape[1], img_size, img_size)
    label_list = torch.zeros(ids.shape[0], 3)
    for i in range(ids.shape[0]):
        for j in range(ids.shape[1]):
            X_training[i, j, :, :] = input_data[ids[i, j], 0, :, :]
            label_list[i, j] = label[ids[i, j]]
    print(X_training.shape,label_list.shape)
    # ids = torch.randint(0, X_training.shape[0], size=(num_training_sample, channel_num))
    # X_stacked = torch.zeros(ids.shape[0], 3 * channel_num, img_size, img_size)
    #
    # for i in range(ids.shape[0]):
    #     for j in range(channel_num):
    #         for k in range(3):
    #             X_stacked[i, j * 3 + k, :, :] = X_training[ids[i, j], k, :, :]
    #
    # Stacked_data = torch.utils.data.TensorDataset(X_stacked, label_list)
    #
    # train_loader_Stacked = torch.utils.data.DataLoader(Stacked_data, batch_size=batch_size, shuffle=True)
    Stacked_data = torch.utils.data.TensorDataset(X_training, label_list)
    dataset_path = '/home/sanaz/Ryerson/Projects/VARNETProject/Stacked_MNIST/data/'
    torch.save(Stacked_data, dataset_path + '/Stacked_data_'+ str(img_size)+'_dataset.pt')
    np.save(dataset_path + '/Labels_'+ str(img_size)+'.npy', label_list)

    return


class JSD:
    def KLD(self, p, q):
        if 0 in q:
            raise ValueError
        return sum(_p * log(_p / _q) for (_p, _q) in zip(p, q) if _p != 0)

    def JSD_core(self, p, q):
        M = [0.5 * (_p + _q) for _p, _q in zip(p, q)]
        return 0.5 * self.KLD(p, M) + 0.5 * self.KLD(q, M)


def evaluation(mini_batch, Total, generator, classifier, img_size, prob):
    num_runs = 1
    split_scores = []
    coorrect = 0
    total = 0
    k_count  = 0
    Total_G_results = torch.tensor([])
    num_modes_list = []
    KLD_list = []
    for num_run in range(num_runs):
        map = {}
        for k1 in range(Total//mini_batch):
            z_ = torch.randn((mini_batch, 100))

            z_ = Variable(z_.cuda())
            G_result = generator(z_)
            splits = 1
            N = len(G_result)
            preds = inception_score(G_result, resize=True)
            for k in range(splits):
                part = preds[k * (N // splits): (k + 1) * (N // splits), :]
                py = np.mean(part, axis=0)
                scores = []
                for i in range(part.shape[0]):
                    pyx = part[i, :]
                    scores.append(entropy(pyx, py))
                split_scores.append(np.exp(np.mean(scores)))
            # if k_count == 0:
            #     Total_G_results = G_result
            #     k_count += 1
            # else:
            #     Total_G_results = torch.cat((Total_G_results, G_result), dim=0)
            # print(G_result.shape)
            predicted_labels = torch.zeros(mini_batch, 3)
            for j in range(3):
                data_dim = G_result[:, j, :, :].reshape(G_result.shape[0], 1, img_size, img_size)
                outputs = classifier(data_dim)
                _, predicted = torch.max(outputs.data, 1)
                # print(predicted.shape)
                predicted_labels[:, j] = predicted
            predicted_labels = predicted_labels.type(torch.LongTensor).cpu().numpy()
            # print(predicted_labels.shape)

            for result in predicted_labels:
                if tuple(result) in list(map.keys()):
                    map[tuple(result)] += 1
                else:
                    map[tuple(result)] = 1
            # print(map.keys(), map.values())
        num_mode = len(map.keys())
        p = np.zeros(1000)
        p = list(map.values())
        p = p / np.sum(p)
        # q = [1.0 / 1000.0] * 1000
        q = prob
        # print(p,q)
        # print(len(split_scores))
        mean1, std1 = np.mean(split_scores), np.std(split_scores)
        print('Inception score', mean1, std1)
        print(' Number of modes %d out of 1000' % (num_mode))
        num_modes_list.append(num_mode)
        print('KL', JSD().KLD(p, q))
        KLD_list.append(JSD().KLD(p, q))
    print('avg kld %f, std %f, mode %d' % (mean(KLD_list), std(KLD_list), mean(num_modes_list)))


def create_sim_data(imgs, type_same):
    '''
    Construct the similar images for VARNET training
    :param imgs: generator output
    :return: The first element of generator ouput is repeated and passed
    '''
    # print(imgs.shape, num)
    if type_same == 0:
        num_var = [1, 2, 4, 5, 10, 20, 25]
        num = num_var[np.random.randint(0, len(num_var) - 1, 1)[0]]
        indexes = np.random.randint(0,imgs.shape[0]-1, num)

    elif type_same == 1:
        num_var = [1, 2, 4, 5, 6]
        num = num_var[np.random.randint(0, len(num_var) - 1, 1)[0]]

        indexes = np.random.randint(0,imgs.shape[0]-1, num)
    elif type_same == 2:
        num_var = [1, 2, 4, 5, 6]
        num = num_var[np.random.randint(0, len(num_var) - 1, 1)[0]]

        indexes = random.sample(list(range(imgs.shape[0])), num)

    fake_data = imgs[indexes, :]
    # print(len(indexes), fake_data.shape, imgs.shape[0],imgs.shape[0]//len(indexes))
    fake_data_repeat = fake_data.repeat(imgs.shape[0]//len(indexes), 1,1,1)
    # print(fake_data_repeat.shape)
    if num == 1:
        label = torch.tensor([0.0])
    else:
        label = torch.tensor([0.51 - (1./ num)])
    # label = label.repeat(imgs.shape[0]//len(indexes), 1)


    # indexes =[0]
    # fake_data = imgs[indexes, :]
    # fake_data = fake_data.repeat(imgs.shape[0]//len(indexes), 1)

    return fake_data_repeat, label


def create_sim_data_labeled(imgs, Label_images, type_same):
    '''
    Construct the similar images for VARNET training
    :param imgs: generator output
    :return: The first element of generator ouput is repeated and passed
    '''
    # print(imgs.shape, num)
    if type_same == 0:
        num_var = [1, 2, 4, 5, 10, 20, 25]
    elif type_same == 1:
        num_var = [1, 2, 4, 5, 6]
    elif type_same == 2:
        num_var = [1, 2, 4, 5, 6]
    num = num_var[np.random.randint(0, len(num_var) - 1, 1)[0]]

    indexes = label_selection(Label_images, num)


    fake_data = imgs[indexes, :]
    # print(len(indexes), fake_data.shape, imgs.shape[0],imgs.shape[0]//len(indexes))
    fake_data_repeat = fake_data.repeat(imgs.shape[0]//len(indexes), 1,1,1)
    # print(fake_data_repeat.shape)
    if num == 1:
        label = torch.tensor([0.0])
    else:
        label = torch.tensor([0.51 - (1./ num)])
    # label = label.repeat(imgs.shape[0]//len(indexes), 1)


    # indexes =[0]
    # fake_data = imgs[indexes, :]
    # fake_data = fake_data.repeat(imgs.shape[0]//len(indexes), 1)

    return fake_data_repeat, label
#
# load_data_create(100000, 32)
# load_data(10, 32)
# load_data_create(100000, 64)

def merge(images, size):
  # print(images.shape, size)
  h, w = images.shape[1], images.shape[2]
  if (images.shape[3] in (3,4)):
    c = images.shape[3]
    img = np.zeros(( h * size[0], w * size[1], c))
    for idx, image in enumerate(images):
      i = idx % size[1]
      j = idx // size[1]
      img[j * h:j * h + h, i * w:i * w + w,:] = image
    return img



def inception_score(imgs, cuda=True, batch_size=32, resize=False, splits=1):
    """Computes the inception score of the generated images imgs
    imgs -- Torch dataset of (3xHxW) numpy images normalized in the range [-1, 1]
    cuda -- whether or not to run on GPU
    batch_size -- batch size for feeding into Inception v3
    splits -- number of splits
    """
    N = len(imgs)

    assert batch_size > 0
    assert N > batch_size

    # Set up dtype
    if cuda:
        dtype = torch.cuda.FloatTensor
    else:
        if torch.cuda.is_available():
            print("WARNING: You have a CUDA device, so you should probably set cuda=True")
        dtype = torch.FloatTensor

    # Set up dataloader
    dataloader = torch.utils.data.DataLoader(imgs, batch_size=batch_size)

    # Load inception model
    inception_model = inception_v3(pretrained=True, transform_input=False).type(dtype)
    inception_model.eval();
    up = nn.Upsample(size=(299, 299), mode='bilinear').type(dtype)
    def get_pred(x):
        # print(x.shape)
        if resize:
            x = up(x)
            # print(x.shape)
        x = inception_model(x)
        # print(x.shape)
        return F.softmax(x).data.cpu().numpy()

    # Get predictions
    preds = np.zeros((N, 1000))

    for i, batch in enumerate(dataloader, 0):
        batch = batch.type(dtype)
        batchv = Variable(batch)
        batch_size_i = batch.size()[0]

        preds[i*batch_size:i*batch_size + batch_size_i] = get_pred(batchv)

    return preds

    # Now compute the mean kl-div



def label_selection(labels, num):
    unique_labels, indices = np.unique(labels, return_index=True)
    # Choose unique designs without repetition
    if num < len(unique_labels):
        indexes = indices[random.sample(range(len(indices)), num)]
    else:
        indexes = random.sample(range(len(labels)), num)

    return indexes
