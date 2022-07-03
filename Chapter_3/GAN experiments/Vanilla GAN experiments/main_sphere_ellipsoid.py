import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import random
from mpl_toolkits.mplot3d import axes3d
from scipy.spatial import distance
import torchvision.transforms as transforms
from torchvision.utils import save_image

import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

from sacred import Experiment
from sacred.observers import FileStorageObserver


from helper import *
from Shape_creator import *
from Models_temp import *
from utils_pre import Logger
from Train import *
from Load_data import *
from Evaluation import check_files_convexity

ex = Experiment('test')
ex.observers.append(FileStorageObserver.create('logs'))

@ex.config
def my_config():

    GAN_model = 'WGAN_GP'#'WGAN_GP'
    mode_collapse = 'PacGAN'
    pac_num = 4
    filled = True
    itr_critic = 5
    dataset_size = 100000 #99906
    size_wanted = 40000
    shape = 'Both' # Both
    generator_activation = 'sigmoid'
    dataset_kind = 'sphere'
    dataset_include_tumor = False
    space_dim = 3
    # dimension = 32
    batch_size = 48 #400
    num_epochs = 1000
    noise_dim = 400
    in_channels_dim = 16
    dimension_orig = [16, 16, 16]  # cube volume
    dimension = [16, 16, 16]
    coef = 0.01
    learning_rate = 0.0002 #0.00002
    learning_rate_g =  0.0002 # 0.0002#0.0002
    learning_rate_d = 0.0002 # 0.0002
    beta = (0.5, 0.5)
    d_thresh = 0.8
    target_points = 1 # Center + r/2 on each axis
    Data_path = '/home/sanaz/Ryerson/Projects/tumorGAN/Data/'
    save_path = '/home/sanaz/Ryerson/Projects/tumorGAN/GAN_simple_3D/'
    step_report = 100
    name_choice = ''
    softSparsity = False
    diversity = False
    connectedFlag = True
@ex.automain
def main(space_dim, mode_collapse, beta, d_thresh, pac_num, name_choice, connectedFlag, dataset_kind,size_wanted, dataset_include_tumor, softSparsity, diversity, step_report, generator_activation, dataset_size, dimension_orig, shape, batch_size, num_epochs, noise_dim,in_channels_dim,
         dimension, coef, learning_rate, learning_rate_g,learning_rate_d,target_points, Data_path, save_path,itr_critic,GAN_model,filled):


    torch.cuda.set_device(1)
    print(torch.cuda.is_available())
    cuda = True if torch.cuda.is_available() else False
    id = ex.current_run._id
    logger = Logger(model_name='DCGAN_{}'.format(id), data_name='Sphere', id=id, image_size=dimension)


    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    string_dimension = [str(int) for int in dimension_orig]
    string_dimension = "_".join(string_dimension)


    dataset_name = 'Dataset_sphere_full_' + string_dimension + '_' + str(dataset_size) + '_' + str(target_points)

    # dataset_name = 'Dataset_iso_sphere_full_' + string_dimension + '_' +  str(dataset_size) + '_' + str(target_points)
    data_file = h5py.File(Data_path + dataset_name + '.h5', 'r')
    Real_data = data_file['data']


    batch_num = int(size_wanted / batch_size)

    if mode_collapse != 'PacGAN':
        pac_num = 1
    generator = Generator(in_channels=in_channels_dim, out_dim=dimension, out_channels=1,
                          noise_dim=noise_dim, activation=generator_activation)
    discriminator = Discriminator(in_channels=int(1 * pac_num), dim=dimension, out_conv_channels=in_channels_dim,model=GAN_model)


    if GAN_model == 'GAN':
        loss = torch.nn.BCELoss()
        train_discriminator = train_discriminator_GAN
        train_generator = train_generator_GAN
    elif GAN_model == 'RSGAN':
        loss = torch.nn.BCEWithLogitsLoss()
        train_discriminator = train_discriminator_RSGAN
        train_generator = train_generator_RSGAN
    elif GAN_model == 'RaSGAN':
        loss = torch.nn.BCEWithLogitsLoss()
        train_discriminator = train_discriminator_RaSGAN
        train_generator = train_generator_RaSGAN
    elif GAN_model == 'WGAN_GP':
        loss = torch.nn.BCELoss()
        train_discriminator = train_discriminator_WGAN_GP
        train_generator = train_generator_WGAN_GP
    # adversarial_loss = torch.nn.BCELoss()

    if cuda:
        generator.cuda()
        discriminator.cuda()
        loss.cuda()
    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=learning_rate_g, betas=beta)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=learning_rate_d, betas = beta)

    for epoch in range(num_epochs):
        print('epoch ', epoch)

        for index in range(batch_num):

            imgs = Real_data[index * batch_size:(index + 1) * batch_size]

            imgs = (imgs) / (target_points)
            real_imgs = imgs.reshape(int((imgs.shape[0])/pac_num), 1*pac_num, dimension[0], dimension[1], dimension[2])
            real_imgs = torch.tensor(real_imgs)
            real_imgs = Variable(real_imgs.type(Tensor))


            valid = Variable(Tensor(real_imgs.shape[0], 1).fill_(1.0), requires_grad=False)
            fake = Variable(Tensor(real_imgs.shape[0], 1).fill_(0.0), requires_grad=False)
            z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], noise_dim))))
           # ---------------------
            #  Train Discriminator
            # ---------------------
            fake_imgs = generator(z)
            fake_imgs = fake_imgs.reshape(int(fake_imgs.shape[0] / pac_num), 1 * pac_num, dimension[0], dimension[1],
                                          dimension[2])
            # # print(fake_imgs.shape, real_imgs.shape)
            d_loss, real_pre, fake_pre_d, real_loss_d, fake_loss_d, d_total_acu, d_sparse1= train_discriminator(discriminator, optimizer_D,
                                                                                         real_imgs,
                                                                                         fake_imgs,
                                                                                         valid, fake, loss,d_thresh, softSparsity)

           

           
            # # -----------------
            #  Train Generator
            # -----------------

            # for g_itr in range(1):
            if (GAN_model == 'WGAN_GP' and index%itr_critic == 0) or GAN_model != 'WGAN_GP':
                # optimizer_G.zero_grad()
                z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], noise_dim))))

                fake_imgs = generator(z)
                fake_imgs = fake_imgs.reshape(int(fake_imgs.shape[0] / pac_num), 1 * pac_num, dimension[0], dimension[1], dimension[2])
                # g_loss, fake_loss_g, fake_pre, d_connect= train_generator(discriminator, optimizer_G, real_imgs,
                #                                                 fake_imgs,
                #                                                 valid, fake, loss,connectedFlag)
                g_loss, fake_loss_g, fake_pre, d_connect, KLD, d_div = train_generator(discriminator, optimizer_G, real_imgs,
                                                                fake_imgs,
                                                                valid, fake, loss, diversity=diversity,target_points=target_points,connectedFlag=connectedFlag)
            if index == batch_num - 1 :
                if softSparsity:
                    print('sparsity',d_sparse1.item())
                if diversity:
                    print('diversity', d_div.item())
               

        if GAN_model == 'GAN':
            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] [D acc: %f] [d connect :%f] [KLD :%f]"
                % (epoch, num_epochs, index, batch_num, d_loss.item(), g_loss.item(), d_total_acu.item(), d_connect, KLD)
            )
        else:
            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]  [d connect :%f] [KLD :%f]"
                % (epoch, num_epochs, index, batch_num, d_loss.item(), g_loss.item(), d_connect, KLD)
            )

        if epoch%step_report ==0 or epoch == num_epochs-1:

            count_z = 100
            if epoch == num_epochs-1:
                count_z = 10000
            z = Variable(Tensor(np.random.normal(0, 1, (count_z, noise_dim))))
            fake_imgs = generator(z)
            dir ='/home/sanaz/Ryerson/Projects/tumorGAN/GAN_simple_3D/results/'

           
            if dataset_include_tumor:
                result_dict = GAN_3D_iso_sphere_full_evaluation(fake_imgs, target_points + 1,dataset_include_tumor, dimension)

            else:
                result_dict = GAN_3D_iso_sphere_full_evaluation(fake_imgs, target_points,dataset_include_tumor,dimension)
            check_files_convexity('', obj=result_dict)
            torch.save(generator, dir+'generator_'+ 'epoch_'+ str(epoch) + 'id_'+str(id)+ '.pt')

            logger.log_metrics(ex, result_dict, epoch)


                
