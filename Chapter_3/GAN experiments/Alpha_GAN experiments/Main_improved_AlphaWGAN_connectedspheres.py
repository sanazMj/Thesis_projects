import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import random
os.environ["GIT_PYTHON_REFRESH"] = "quiet"

from mpl_toolkits.mplot3d import axes3d
from scipy.spatial import distance
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable


import torch.nn as nn
import torch.nn.functional as F
import torch
from helper import *
from Shape_creator import *
from sacred import Experiment
from sacred.observers import FileStorageObserver

from Model_alphaWGAN import *
from utils_pre import Logger
from Train import *
from Load_data import *

from Evaluation import check_files_convexity
ex = Experiment('test')
ex.observers.append(FileStorageObserver.create('logs'))

def calc_gradient_penalty(model, x, x_gen, w=10):
    LAMBDA = 10
    _eps = 1e-15
    """WGAN-GP gradient penalty"""
    assert x.size()==x_gen.size(), "real and sampled sizes do not match"
    alpha_size = tuple((len(x), *(1,)*(x.dim()-1)))
    alpha_t = torch.cuda.FloatTensor if x.is_cuda else torch.Tensor
    alpha = alpha_t(*alpha_size).uniform_()
    x_hat = x.data*alpha + x_gen.data*(1-alpha)
    x_hat = Variable(x_hat, requires_grad=True)

    def eps_norm(x):
        x = x.view(len(x), -1)
        return (x*x+_eps).sum(-1).sqrt()
    def bi_penalty(x):
        return (x-1)**2

    grad_xhat = torch.autograd.grad(model(x_hat).sum(), x_hat, create_graph=True, only_inputs=True)[0]

    penalty = w*bi_penalty(eps_norm(grad_xhat)).mean()
    return penalty

@ex.config
def my_config():

    GAN_model = 'WGAN_GP'
    filled = True
    pac_num = 1
    itr_critic = 5
    dataset_size = 100000 #99906
    size_wanted = 40000
    shape = 'Both' # Both
    generator_activation = 'sigmoid'
    dataset_kind = 'Matlab'
    dataset_include_tumor = False
    space_dim = 3
    # dimension = 32
    batch_size = 50

    num_epochs = 500
    noise_dim = 500
    latent_dim = noise_dim

    in_channels_dim = 16
    dimension_orig = [16, 16, 16]  # cube volume
    dimension = [16, 16, 16]
    coef = 0.01
    learning_rate = 0.0002 #0.00002
    learning_rate_g = 0.0002#0.0002
    learning_rate_d = 0.0002
    target_points = 1 # Center + r/2 on each axis
    Data_path = '/home/sanaz/Ryerson/Projects/tumorGAN/Data/'
    save_path = '/home/sanaz/Ryerson/Projects/tumorGAN/GAN_simple_3D/'
    step_report = 100
    name_choice = ''
    sofSparsity = False
    diversity = True
    connectivity=True # If set to true adds the connectivity loss to the entire loss as well
    cuda_device = 0
@ex.automain
def main(space_dim, cuda_device, name_choice, latent_dim, connectivity, pac_num, dataset_kind,size_wanted, dataset_include_tumor, sofSparsity, diversity, step_report, generator_activation, dataset_size, dimension_orig, shape, batch_size, num_epochs, noise_dim,in_channels_dim,
         dimension, coef, learning_rate, learning_rate_g,learning_rate_d,target_points, Data_path, save_path,itr_critic,GAN_model,filled):


    torch.cuda.set_device(cuda_device)
    print(torch.cuda.is_available())
    cuda = True if torch.cuda.is_available() else False

    id = ex.current_run._id
    logger = Logger(model_name='DCGAN_{}'.format(id), data_name='Sphere', id=id, image_size=dimension)

    string_dimension = [str(int) for int in dimension_orig]
    string_dimension = "_".join(string_dimension)
    if dataset_kind == 'Matlab':
        dataset_name = 'Dataset_Feb_28_March_01_limited_16_one_tumor_edited'
    else:
        dataset_name = 'Dataset_sphere_full_16_16_16_100000_1'
    # dataset_name = 'Dataset_' + dataset_kind +'_'+ string_dimension + '_' + str(dataset_size) + '_' + str(target_points)
    data_file = h5py.File(Data_path + dataset_name + '.h5', 'r')


    Real_data = data_file['data']
    batch_num = int(size_wanted / batch_size)


    G = Generator(channel=2, noise=latent_dim)
    CD = Code_Discriminator(code_size=latent_dim, num_units=4096)
    D = Discriminator_with_discrete_levels_using_inception(channel=32, target_point=target_points + int(dataset_include_tumor),
                                                      is_dis=True)
    E = Discriminator(channel=32, out_class=latent_dim, is_dis=False)

    G.cuda()
    D.cuda()
    CD.cuda()
    E.cuda()

    # %%

    g_optimizer = optim.Adam(G.parameters(), lr=0.0002)
    d_optimizer = optim.Adam(D.parameters(), lr=0.0002)
    e_optimizer = optim.Adam(E.parameters(), lr=0.0002)
    cd_optimizer = optim.Adam(CD.parameters(), lr=0.0002)

    criterion_bce = nn.BCELoss().cuda()
    criterion_l1 = nn.L1Loss().cuda()
    criterion_mse = nn.MSELoss().cuda()
    g_iter = 1
    d_iter = 1
    cd_iter = 1
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    for epoch in range(num_epochs):
        print('epoch ', epoch)

        for index in range(batch_num):

            for p in D.parameters():
                p.requires_grad = False
            for p in CD.parameters():
                p.requires_grad = False
            for p in E.parameters():
                p.requires_grad = True
            for p in G.parameters():
                p.requires_grad = True
            ###############################################
            # Train Encoder - Generator
            ###############################################
            G.zero_grad()
            E.zero_grad()
            imgs = Real_data[index * batch_size:(index + 1) * batch_size]
            # imgs = batch
            if dataset_include_tumor:
                imgs = (imgs) / (target_points+1)
            else:
                imgs = (imgs) / (target_points)
            real_imgs = imgs.reshape(int((imgs.shape[0]) / pac_num), 1 * pac_num, dimension[0], dimension[1], dimension[2])
            real_imgs = torch.tensor(real_imgs)
            real_images = Variable(real_imgs.type(Tensor))
            _batch_size = real_images.size(0)
            z_rand = Variable(torch.randn((_batch_size, latent_dim)), requires_grad=False).cuda()
            # print('z_rand', z_rand)
            z_hat = E(real_images).view(_batch_size, -1)
            # print('z_hat', z_hat)
            x_hat = G(z_hat)
            # print('x_hat', x_hat)
            x_rand = G(z_rand)
            if connectivity:
                Loss_conncted = connected(x_rand, target_points+int(dataset_include_tumor)) + connected(x_hat, target_points+int(dataset_include_tumor))

            # print('x_rand', x_rand)
            # c_loss = -CD(z_hat).mean()
            c_loss = -CD(z_hat.detach()).mean()
            # print('c_loss', c_loss)

            d_real_loss = D(x_hat).mean()
            d_fake_loss = D(x_rand).mean()
            d_loss = -d_fake_loss - d_real_loss
            l1_loss = 10 * criterion_l1(x_hat, real_images)
            if connectivity:
                loss1 = l1_loss + c_loss + d_loss + Loss_conncted
            else:
                loss1 = l1_loss + c_loss + d_loss
            # print(l1_loss, c_loss, d_loss)
            # print('loss1', loss1)
            # if iters < g_iter - 1:
            #     loss1.backward(retain_graph=True)
            # else:
            loss1.backward(retain_graph=True)
            # print('c_loss', c_loss)

            e_optimizer.step()
            g_optimizer.step()
            g_optimizer.step()

            ###############################################
            # Train D
            ###############################################
            for p in D.parameters():
                p.requires_grad = True
            for p in CD.parameters():
                p.requires_grad = False
            for p in E.parameters():
                p.requires_grad = False
            for p in G.parameters():
                p.requires_grad = False

            for iters in range(d_iter):
                d_optimizer.zero_grad()
                z_hat = E(real_images).view(_batch_size, -1)

                z_rand = Variable(torch.randn((_batch_size, latent_dim))).cuda()
                # print('z_rand', z_rand)
                # z_hat = E(real_images).view(_batch_size, -1)
                # print('z_hat', z_hat)
                x_hat = G(z_hat)
                x_rand = G(z_rand)
                x_loss2 = -2 * D(real_images).mean() + D(x_hat).mean() + D(x_rand).mean()
                gradient_penalty_r = calc_gradient_penalty(D, real_images.data, x_rand.data)
                gradient_penalty_h = calc_gradient_penalty(D, real_images.data, x_hat.data)

                loss2 = x_loss2 + gradient_penalty_r + gradient_penalty_h
                # print('loss2', loss2)

                loss2.backward(retain_graph=True)
                d_optimizer.step()

                ###############################################
                # Train CD
                ###############################################
            for p in D.parameters():
                p.requires_grad = False
            for p in CD.parameters():
                p.requires_grad = True
            for p in E.parameters():
                p.requires_grad = False
            for p in G.parameters():
                p.requires_grad = False

            for iters in range(cd_iter):
                # print('z_hat', z_hat)

                cd_optimizer.zero_grad()
                # z_hat = E(real_images).view(_batch_size, -1)
                z_rand = Variable(torch.randn((_batch_size, latent_dim))).cuda()
                gradient_penalty_cd = calc_gradient_penalty(CD, z_hat.data, z_rand.data)
                # print('c_loss', c_loss)
                loss3 = -CD(z_rand).mean() - c_loss + gradient_penalty_cd
                # print('loss3', loss3, gradient_penalty_cd)
                loss3.backward(retain_graph=True)
                cd_optimizer.step()
        if connectivity:
            print('[{}/{}]'.format(epoch, num_epochs),
                  'D: {:<8.3}'.format(loss2.item()),
                  'En_Ge: {:<8.3}'.format(loss1.item()),
                  'Code: {:<8.3}'.format(loss3.item()),
                  'Connected loss: {:<8.3}'.format(Loss_conncted.item())
                  )
        else:
            print('[{}/{}]'.format(epoch, num_epochs),
                  'D: {:<8.3}'.format(loss2.item()),
                  'En_Ge: {:<8.3}'.format(loss1.item()),
                  'Code: {:<8.3}'.format(loss3.item()))

        if epoch % step_report == 0 or epoch == num_epochs - 1:
            tumor_count = []
            OAR_count = []
            tumor_sizes = []
            OAR_sizes = []
            count_z = 100
            if epoch == num_epochs - 1 or epoch>90:
                count_z = 10000
            z = Variable(Tensor(np.random.normal(0, 1, (count_z, noise_dim))))
            fake_imgs = G(z)
            dir = '/home/sanaz/Ryerson/Projects/tumorGAN/GAN_simple_3D/results/'


            if dataset_include_tumor:
                result_dict = GAN_3D_iso_sphere_full_evaluation(fake_imgs, target_points + 1,dataset_include_tumor, dimension)

            else:
                result_dict = GAN_3D_iso_sphere_full_evaluation(fake_imgs, target_points,dataset_include_tumor,dimension)
            check_files_convexity('', name_data=dataset_kind, obj=result_dict)
            torch.save(G, dir + 'generator_' + 'epoch_' + str(epoch) + 'id_' + str(id) + '.pt')
            logger.log_metrics(ex, result_dict, epoch)
            
