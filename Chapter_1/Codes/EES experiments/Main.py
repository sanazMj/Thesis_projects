import time
import torch
import numpy as np
from torch.autograd.variable import Variable
from sacred import Experiment
from sacred.observers import FileStorageObserver
from Preprocessing import read_ees_dataset
from model import initialize_models
from Utils.utils_def import *
from Utils.utils_log import *
from Models.Minibatch_discrimination import *
ex = Experiment('test')
ex.observers.append(FileStorageObserver.create('logs'))

@ex.config
def my_config():
    Model = 'Normal'
    Model_structure = 'Convolutional' #['FF','Convolutional', 'ConvOriginal']
    Model_type = 'Conditional' #['Conditional' 'Vanilla']
    mode_collapse = '' #'DSGAN, ' 'Minibatch', PacGAN
    num_epochs = 50
    categorization = 8
    full_image = False
    partial_2fold = True
    Quarter_fill = False
    dataset_Balance = False
    prepadding = False
    prepadding_width = 1
    Losses = ['BCE']
    Pixel_Full = 9
    Pixel_Partial = (Pixel_Full // 2) + 1
    Minibatch = ( mode_collapse == 'Minibatch')
    minibatch_kind = 'L1 Norm'
    n_critic = 1
    zdim = 100
    batch_size = 100
    channels = 512
    Kernel_factor = 1
    Channel_factor = 1
    Pack_num = 4
    ndf = 2048 # Num Discriminator Features
    ngf = 512 # Num Generator Features
    compare = False # compare 8 cat wiht 2 cat with same test_noise
@ex.automain
def main(batch_size,channels,n_critic,compare,
         zdim, num_epochs, categorization, full_image,partial_2fold,
         Pixel_Full, Pixel_Partial, Model, Model_structure, Model_type, dataset_Balance,Quarter_fill,
         prepadding, prepadding_width, Kernel_factor, Channel_factor, Losses, Pack_num, Minibatch, minibatch_kind,mode_collapse, ngf, ndf):

    torch.cuda.set_device(1)

    id = ex.current_run._id
    logger = Logger(model_name='Conditional_DCGAN_{}'.format(id), data_name='EES', id=id, image_size=full_image,
                    model_type=Model)

    if mode_collapse == 'PacGAN':
        Pack_number = Pack_num
    else:
        Pack_number = 1

    if Model in ['Normal', 'Complex'] :
        from train import train_discriminator, train_generator
    if Pixel_Full == 9 and partial_2fold == True and Model_type == 'Vanilla'and Model_structure == 'Convolutional':
        from Models.Partial_image.len_9.Model_9_Partial_CGAN_kernel3 import initialize_models
    elif Pixel_Full == 9 and partial_2fold == True and Model_type == 'Conditional' and Model_structure == 'Convolutional':
        from Models.Partial_image.len_9.Model_9_Partial_cCGAN_kernel3 import initialize_models
    if Pixel_Full == 9 and partial_2fold == True and Model_type == 'Vanilla'and Model_structure == 'ConvOriginal':
        from Models.Partial_image.len_9.Model_9_CGAN_Original import initialize_models
    elif Pixel_Full == 9 and partial_2fold == True and Model_type == 'Conditional' and Model_structure == 'ConvOriginal':
        from Models.Partial_image.len_9.Model_9_cCGAN_Original import initialize_models
    elif Pixel_Full == 9 and partial_2fold == True and Model_type == 'Vanilla'and Model_structure == 'FF':
        from Models.Partial_image.len_9.Model_9_Partial_GAN import initialize_models
    elif Pixel_Full == 9 and partial_2fold == True and Model_type == 'Conditional' and Model_structure == 'FF':
        from Models.Partial_image.len_9.Model_9_Partial_cGAN import initialize_models

    elif Pixel_Full == 19 and partial_2fold ==  True and Model_type=='Conditional':
        from Models.Partial_image.len_19.Model_19_Partial_Type1 import initialize_models
    elif Pixel_Full == 19 and partial_2fold ==  True and Model_type=='Vanilla':
        from Models.Partial_image.len_19.Model_19_partial_type1_CGAN import initialize_models

    data_loader, dataset, pixel_to_label, cat_names = read_ees_dataset( batch_size=batch_size,
                                                                        categorization=categorization,
                                                                        full_image=full_image,
                                                                        partial_2fold=partial_2fold,
                                                                        Pixel_Full=Pixel_Full,
                                                                        Quarter_fill=Quarter_fill,
                                                                        prepadding=prepadding,
                                                                        prepadding_width=prepadding_width,
                                                                        lookup=True, balance=dataset_Balance)
    args = [channels, Channel_factor, Kernel_factor, zdim,  dataset['num_features'], dataset['num_condition'], dataset['num_pixels']]

    if mode_collapse == 'Minibatch' and Model_structure == 'FF':
        minibatch_net = MinibatchDiscrimination(ndf//4, ndf//8, 5, Minibatch_kind='L1 Norm')
        minibatch_net.cuda()


    if Pixel_Full == 19 or (Pixel_Full == 9 and (Model_structure == 'Convolutional' or Model_structure =='ConvOriginal')):

        generator, discriminator, g_optimizer, d_optimizer, loss_BCE = initialize_models(channels, Channel_factor, Kernel_factor, zdim, dataset['num_condition'],
                                                                                         dataset['num_pixels'], Minibatch, minibatch_kind, Pack_number)
        loss_dict = {}
        loss_dict['BCE'] = loss_BCE
    elif Pixel_Full == 9 and Model_structure == 'FF':
        generator, discriminator, g_optimizer, d_optimizer, loss_BCE = initialize_models(dataset['num_features'], dataset['num_condition'], dataset['num_pixels'], ndf, ngf, zdim, minibatch_net)
        loss_dict = {}
        loss_dict['BCE'] = loss_BCE
    else:


        generator, discriminator, g_optimizer, d_optimizer, loss_dict = initialize_models(Model, Model_structure, Model_type, full_image, Pixel_Full, Losses, args)


    if compare:
        test_noise, labels_of_test_noise = create_test_samples(categorization, 2, False, zdim, test_samples_per_category=500)

    else:
        test_noise, labels_of_test_noise = create_test_samples(dataset['num_condition'], dataset['num_condition'], False, zdim, test_samples_per_category=500)
    if full_image:
        pixel = Pixel_Full
    else:
        pixel = Pixel_Partial

    # num_epochs = 1
    for epoch in range(num_epochs):
        print('epoch:', epoch)
        class_accuracies = {}
        total_unknowns_Hp = 0
        total_unknowns_Lp = 0
        accuracy_matrix = np.zeros((dataset['num_condition'], 2))  # Number of unique labels x 2

        for n_batch, (real_batch, labels) in enumerate(data_loader):
            batch_labels_reshape = labels
            if mode_collapse == 'PacGAN':
                real_batch = real_batch.reshape(int(real_batch.shape[0] / Pack_number),  real_batch.shape[1]* Pack_number)
                batch_labels_reshape = labels.reshape(int(labels.shape[0] / Pack_number),  labels.shape[1]* Pack_number)

            if mode_collapse == 'DSGAN':
                real_batch = torch.cat((real_batch, real_batch), dim=0)
                B = int(real_batch.size(0) / 2)
                labels = torch.cat((labels, labels), dim=0)
                batch_labels_reshape = labels

            if torch.cuda.is_available():
                real_data = real_batch.cuda()
                labels_List = labels.cuda()
            else:
                real_data = real_batch
                labels_List = labels
            if Model_structure == 'FF':
                Disc_Labels = labels_List
            else:
                Disc_Labels = torch.tensor(
                    np.stack([np.tile(labels_List[i].cpu(), (pixel, pixel, 1)) for i in range(labels_List.shape[0])]))
            if torch.cuda.is_available(): Disc_Labels = Disc_Labels.cuda()



            fake_data = generator(noise(int(real_data.size(0)*Pack_number),zdim), labels_List).detach()
            if mode_collapse == 'PacGAN' and (Model_structure == 'Convolutional' or Model_structure == 'ConvOriginal'):

                fake_data = fake_data.reshape(int(fake_data.shape[0] / Pack_number),
                                                fake_data.shape[1] * Pack_number, fake_data.shape[2], fake_data.shape[3])
                Disc_Labels = Disc_Labels.reshape(int(Disc_Labels.shape[0] / Pack_number),
                                                Disc_Labels.shape[1] , Disc_Labels.shape[2], Disc_Labels.shape[3]* Pack_number)
            elif mode_collapse == 'PacGAN' and Model_structure == 'FF':
                fake_data = fake_data.reshape(int(fake_data.shape[0] / Pack_number),
                                                fake_data.shape[1] * Pack_number)
                                                   real_data, fake_data, labels_List)
            d_error, d_pred_real, d_pred_fake = train_discriminator(discriminator, loss_dict, d_optimizer,
                                                                    real_data, fake_data, Disc_Labels)

            # 2. Train Generator
            # Generate fake data
            #
            for j in range(n_critic):
                z_noise = noise(int(real_data.size(0)*Pack_number),zdim)
                fake_data = generator(z_noise, labels_List)
                if mode_collapse == 'PacGAN' and (Model_structure == 'Convolutional' or Model_structure == 'ConvOriginal'):

                    fake_data = fake_data.reshape(int(fake_data.shape[0] / Pack_number),
                                                  fake_data.shape[1] * Pack_number, fake_data.shape[2],
                                                  fake_data.shape[3])
                elif mode_collapse == 'PacGAN' and Model_structure == 'FF':
                    fake_data = fake_data.reshape(int(fake_data.shape[0] / Pack_number),
                                                  fake_data.shape[1] * Pack_number)
                # Train G
                g_error = train_generator(discriminator, loss_dict, g_optimizer, fake_data, z_noise, Disc_Labels, mode_collapse, 0)
               
        if epoch % 40 == 0 and epoch>1:
                print('epoch 40%0')
                save_generation_distribution(ex, epoch, categorization, cat_names, generator, pixel_to_label,
                                             dataset['num_condition'], full_image, pixel, Quarter_fill, zdim,
                                             plot_generated_distribution=True)
        if epoch == num_epochs-1:
                print('training finished')
                if epoch == num_epochs - 1 and pixel_to_label:
                    save_generation_distribution(ex, epoch, categorization, cat_names, generator, pixel_to_label,
                                                        dataset['num_condition'], full_image, pixel, Quarter_fill,zdim,
                                                        plot_generated_distribution=True)

