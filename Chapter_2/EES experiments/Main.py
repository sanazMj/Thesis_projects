from sacred import Experiment
from sacred.observers import FileStorageObserver
from EES.Main.Preprocessing import read_ees_dataset
from Utils.utils_def import *
from Utils.utils_log import *
from Models.Minibatch_discrimination import *
from Utils.utils_def import ModelSwitcher
from Training.train import train_discriminator, train_generator, train_varmeter

ex = Experiment('test')
ex.observers.append(FileStorageObserver.create('logs'))


@ex.config
def my_config():
    model = 'Normal'
    model_structure = 'Convolutional'  # ['FF','Convolutional', 'ConvOriginal']
    model_type = 'Vanilla'  # ['Conditional' 'Vanilla']
    mode_collapse = 'VarGAN'  # 'DSGAN, ' 'Minibatch', PacGAN, VarGAN
    num_epochs = 1
    categorization = 2.3
    full_image = False
    partial_2fold = True
    Quarter_fill = False
    dataset_Balance = False
    prepadding = False
    prepadding_width = 1
    Losses = ['BCE']
    Pixel_Full = 19
    Pixel_Partial = (Pixel_Full // 2) + 1
    Minibatch = (mode_collapse == 'Minibatch')
    minibatch_kind = 'L1 Norm'
    n_critic = 1
    zdim = 100
    batch_size = 512
    channels = 512
    Kernel_factor = 1
    Channel_factor = 1
    Pack_number = batch_size  # VARGAN
    PacGAN_pacnum = 1  # PAcGAN
    var_coef = 1
    ndf = 2048  # Num Discriminator Features
    ngf = 512  # Num Generator Features
    same_creation_type = 3
    num_hidden_layers = 2
    Level_line = 1
    Slope = 10
    Coef = 10
    n_mixture = 512
    gen_num = 0

@ex.automain
def main(batch_size, channels, n_critic,
         zdim, num_epochs, categorization, full_image, partial_2fold,
         Pixel_Full, Pixel_Partial, model, model_structure, model_type, dataset_Balance, Quarter_fill,
         same_creation_type,
         prepadding, prepadding_width,
         Kernel_factor, num_hidden_layers, PacGAN_pacnum, Channel_factor, Losses, Pack_number, Minibatch,
         minibatch_kind, mode_collapse, gen_num,  ngf, ndf, var_coef,
         Level_line, Slope, Coef, n_mixture):
    torch.cuda.set_device(0)
    print(torch.cuda.is_available())
    id = ex.current_run._id
    logger = Logger(model_name='DCGAN_{}'.format(id), data_name='EES', id=id, image_size=full_image,
                    model_type=model)

    # if mode_collapse == 'PacGAN' or mode_collapse == 'VarGAN':
    #     Pack_number = Pack_num
    # else:
    #     Pack_number = 1

    # if Model in ['Normal', 'Complex']:
    #     from Training.train import train_discriminator, train_generator, train_varmeter
    # if mode_collapse == 'Minibatch' and model_structure == 'FF':
    #     minibatch_net = MinibatchDiscrimination(ndf // 4, ndf // 8, 5, Minibatch_kind='L1 Norm')
    #     minibatch_net.cuda()
    # if mode_collapse != 'Minibatch':
    #     minibatch_net = False
    pixel = Pixel_Full if full_image else Pixel_Partial
    minibatch_net = False if mode_collapse != 'Minibatch' else True

    data_loader, dataset, pixel_to_label, cat_names = read_ees_dataset(batch_size=batch_size,
                                                                       categorization=categorization,
                                                                       full_image=full_image,
                                                                       partial_2fold=partial_2fold,
                                                                       Pixel_Full=Pixel_Full,
                                                                       Quarter_fill=Quarter_fill,
                                                                       prepadding=prepadding,
                                                                       prepadding_width=prepadding_width,
                                                                       lookup=True, balance=dataset_Balance)

    args = [channels, Channel_factor, Kernel_factor, zdim, dataset['num_features'], dataset['num_condition'],
            dataset['num_pixels'], Minibatch, minibatch_kind, Pack_number, PacGAN_pacnum, ndf, ngf, minibatch_net, mode_collapse, gen_num]

    switcher_model = ModelSwitcher(args)
    generator, discriminator, varmeter, g_optimizer, d_optimizer, varmeter_optimizer, loss = switcher_model.condition_to_model(
        Pixel_Full, partial_2fold, model_type, model_structure)

    loss_dict = {'BCE': loss}
    test_noise, labels_of_test_noise = create_test_samples(dataset['num_condition'], dataset['num_condition'], False,
                                                           zdim, test_samples_per_category=500)

    for epoch in range(num_epochs):

        for n_batch, (real_batch, labels) in enumerate(data_loader):
            if real_batch.shape[0] != batch_size:
                break
            n_data_shape = real_batch.shape[0]
            real_data1 = real_batch
            batch_labels_reshape = labels

            real_data1 = real_data1.reshape(int(real_data1.shape[0] / PacGAN_pacnum),
                                            real_data1.shape[1] * PacGAN_pacnum)
            batch_labels_reshape = labels.reshape(int(labels.shape[0] / PacGAN_pacnum), labels.shape[1] * PacGAN_pacnum)

            if mode_collapse == 'DSGAN':
                real_batch = torch.cat((real_batch, real_batch), dim=0)
                B = int(real_batch.size(0) / 2)
                labels = torch.cat((labels, labels), dim=0)
                batch_labels_reshape = labels
            if mode_collapse == 'VarGAN':
                # print(labels.shape[0])
                batch_labels_reshape = labels.reshape(int(labels.shape[0] // Pack_number),
                                                      labels.shape[1] * Pack_number)

                if model_structure == 'Convolutional' or model_structure == 'ConvOriginal':
                    # batch_VAR_label = labels[0, :]
                    # batch_VAR_label = torch.tensor(np.stack([np.tile(batch_VAR_label.cpu(), (pixel, pixel, 1)) for i in range(labels.shape[0])]))
                    # print(batch_VAR_label.shape)
                    # batch_VAR_label = batch_VAR_label.reshape(int(batch_VAR_label.shape[0] / Pack_number),
                    #                                            batch_VAR_label.shape[1], batch_VAR_label.shape[2],
                    #                                            batch_VAR_label.shape[3] * Pack_number)
                    #
                    #
                    # batch_VAR = real_batch[0, :]
                    # batch_VAR = batch_VAR.repeat(real_batch.shape[0], 1)
                    # batch_VAR = batch_VAR.reshape(int(batch_VAR.shape[0] / Pack_number),
                    #                               batch_VAR.shape[1] * Pack_number)
                    # Create similar data using sim function
                    # print(real_batch.shape, labels.shape)
                    batch_VAR, batch_VAR_label, label_target = create_sim_data(real_batch, pixel,
                                                                               batch_size, labels, n_mixture,
                                                                               type=same_creation_type,
                                                                               Level_line=Level_line, Coef=Coef,
                                                                               slope=Slope)

                    batch_VAR = batch_VAR.reshape(int(batch_VAR.shape[0] / Pack_number),
                                                  batch_VAR.shape[1] * Pack_number)
                    batch_VAR = batch_VAR.cuda()

                    batch_VAR_label = batch_VAR_label.reshape(int(batch_VAR_label.shape[0] / Pack_number),
                                                              batch_VAR_label.shape[1], batch_VAR_label.shape[2],
                                                              batch_VAR_label.shape[3] * Pack_number)

                    batch_VAR_label = batch_VAR_label.cuda()
                    # print(batch_VAR.shape, batch_VAR_label.shape)

                    label_target = label_target.repeat(batch_VAR.shape[0], 1)
                    label_target = label_target.cuda()
                    ## end of Creating similar data using sim function

                else:

                    batch_VAR_label = labels[0, :]
                    batch_VAR_label = batch_VAR_label.repeat(real_batch.shape[0], 1)
                    batch_VAR_label = batch_VAR_label.reshape(int(batch_VAR_label.shape[0] / Pack_number),
                                                              batch_VAR_label.shape[1] * Pack_number)

                    batch_VAR = real_batch[0, :]
                    batch_VAR = batch_VAR.repeat(real_batch.shape[0], 1)
                    batch_VAR = batch_VAR.reshape(int(batch_VAR.shape[0] / Pack_number),
                                                  batch_VAR.shape[1] * Pack_number)

                real_batch = real_batch.reshape(int(real_batch.shape[0] / Pack_number),
                                                real_batch.shape[1] * Pack_number)
            real_data = Variable(real_batch)

            if torch.cuda.is_available():
                real_data = real_data.cuda()
                real_data1 = real_data1.cuda()
                labels = labels.cuda()
                batch_labels_reshape = batch_labels_reshape.cuda()
                batch_VAR = batch_VAR.cuda()
                batch_VAR_label = batch_VAR_label.cuda()
                labels_List = labels.cuda()

            else:
                real_data = real_batch
                labels_List = labels
            # print(real_data.shape, labels_List.shape)
            if model_structure == 'FF':
                Disc_Labels = labels_List
            else:
                Disc_Labels = torch.tensor(
                    np.stack([np.tile(labels_List[i].cpu(), (pixel, pixel, 1)) for i in range(labels_List.shape[0])]))
            if torch.cuda.is_available(): Disc_Labels = Disc_Labels.cuda()

            # print(real_data.shape)
            fake_data = generator(noise(int(n_data_shape), zdim), labels_List).detach()
            # print(fake_data.shape)
            if (model_structure == 'Convolutional' or model_structure == 'ConvOriginal'):
                # print(fake_data.shape)
                fake_data = fake_data.reshape(int(fake_data.shape[0] / PacGAN_pacnum),
                                              fake_data.shape[1] * PacGAN_pacnum, fake_data.shape[2],
                                              fake_data.shape[3])
                Disc_Labels_PacGAN = Disc_Labels.reshape(int(Disc_Labels.shape[0] / PacGAN_pacnum),
                                                         Disc_Labels.shape[1], Disc_Labels.shape[2],
                                                         Disc_Labels.shape[3] * PacGAN_pacnum)
            elif model_structure == 'FF':
                fake_data = fake_data.reshape(int(fake_data.shape[0] / PacGAN_pacnum),
                                              fake_data.shape[1] * PacGAN_pacnum)

            if mode_collapse == 'VarGAN' and model_structure != 'FF':
                Disc_Labels_VARGAN = Disc_Labels.reshape(int(Disc_Labels.shape[0] / Pack_number),
                                                         Disc_Labels.shape[1], Disc_Labels.shape[2],
                                                         Disc_Labels.shape[3] * Pack_number)
            elif mode_collapse == 'VarGAN' and model_structure == 'FF':

                Disc_Labels_VARGAN = Disc_Labels.reshape(int(Disc_Labels.shape[0] / Pack_number),
                                                         Disc_Labels.shape[1] * Pack_number)

            # print(time.time())
            # fake_data = generator(noise(real_data.size(0), zdim), labels_List)
            # Train D

            # print(time.time())
            # d_error, d_pred_real, d_pred_fake = train_discriminator(discriminator,loss_d,d_optimizer,
            #                                                         real_data, fake_data, labels_List)
            # print('Disc', real_data1.shape,fake_data.shape,Disc_Labels_PacGAN.shape)
            d_error, d_pred_real, d_pred_fake = train_discriminator(discriminator, loss, d_optimizer,
                                                                    real_data1, fake_data, Disc_Labels_PacGAN)
            # print('Disc', batch_VAR_label.shape, Disc_Labels_reshaped.shape)

            var_error, var_pred_real, var_pred_fake = train_varmeter(varmeter, loss, varmeter_optimizer,
                                                                     real_data, batch_VAR, Disc_Labels_VARGAN,
                                                                     batch_VAR_label, label_target)
            # 2. Train Generator
            # Generate fake data
            #
            for j in range(n_critic):
                z_noise = noise(int(n_data_shape), zdim)
                fake_data = generator(z_noise, labels_List)
                if (model_structure == 'Convolutional' or model_structure == 'ConvOriginal'):
                    # print(fake_data.shape)
                    fake_data_PacGAN = fake_data.reshape(int(fake_data.shape[0] / PacGAN_pacnum),
                                                         fake_data.shape[1] * PacGAN_pacnum, fake_data.shape[2],
                                                         fake_data.shape[3])
                elif model_structure == 'FF':
                    fake_data_PacGAN = fake_data.reshape(int(fake_data.shape[0] / PacGAN_pacnum),
                                                         fake_data.shape[1] * PacGAN_pacnum)
                if mode_collapse == 'VarGAN' and (
                        model_structure == 'Convolutional' or model_structure == 'ConvOriginal'):
                    fake_data_VARGAN = fake_data.reshape(int(fake_data.shape[0] / Pack_number),
                                                         fake_data.shape[1] * Pack_number, fake_data.shape[2],
                                                         fake_data.shape[3])

                # Train G
                g_error, gprediction_d, gprediction_var = train_generator(discriminator, varmeter, loss, g_optimizer,
                                                                          fake_data_PacGAN, z_noise, fake_data_VARGAN,
                                                                          Disc_Labels_PacGAN, Disc_Labels_VARGAN,
                                                                          mode_collapse, var_coef)
                # print('train_g')

            # fake_data = generator(noise(real_batch.size(0)), labels_List)
            # Train G
            # print(time.time())
            # g_error = train_generator(discriminator,loss_g, g_optimizer, fake_data, labels_List)
            #         print('train g,',time.time())
            # Log error

            # Log error
            # print(time.time())
            logger.log(d_error, g_error, epoch, n_batch, dataset['num_batches'])

            # Model Checkpoints
            #         logger.save_models(generator, discriminator, epoch)

            # Check the test samples
        # test_images = generator(test_noise, labels_of_test_noise).data.cpu().numpy()
        # test_images = np.where(test_images > 0.5, 1, 0)  # convert to binary output
        # # test_images = np.where(test_images > 0, 1, 0)  # convert to binary output
        #
        # conditioned_labels, lookup_labels, lookup_labels_quad, num_unknowns = predict_test_samples(
        #     test_images,
        #     labels_of_test_noise,
        #     categorization,
        #     pixel_to_label,
        #     full_image, pixel, Quarter_fill)
        # accuracy_matrix, class_accuracies, accuracy_matrix_quad, class_accuracies_quad, accuracy_matrix_without_unknowns, class_accuracies_without_unknowns, accuracy_matrix_without_unknowns_quad, class_accuracies_without_unknowns_quad = create_accuracy_matrix(
        #     dataset['num_condition'], conditioned_labels,
        #     lookup_labels, lookup_labels_quad, cat_names)
        if epoch % 40 == 0 and epoch > 1:
            save_generation_distribution(ex, epoch, categorization, cat_names, generator, pixel_to_label,
                                         dataset['num_condition'], full_image, pixel, Quarter_fill, zdim,
                                         plot_generated_distribution=False)
            # evaluate_generation_distribution(ex, epoch, test_images, categorization, cat_names, generator,
            #                                  pixel_to_label,
            #                                  dataset['num_condition'], full_image, pixel, Quarter_fill, zdim,
            #                                  accuracy_matrix, class_accuracies, accuracy_matrix_quad,
            #                                  class_accuracies_quad, accuracy_matrix_without_unknowns,
            #                                  accuracy_matrix_without_unknowns_quad,
            #                                  conditioned_labels, lookup_labels, lookup_labels_quad, num_unknowns,
            #                                  plot_generated_distribution=False)

        logger.display_status(epoch, num_epochs, n_batch, dataset['num_batches'], d_error, g_error, var_error,
                              d_pred_real,
                              d_pred_fake, var_pred_real, var_pred_fake, gprediction_d, gprediction_var)

        # if n_batch % dataset['num_batches'] == 0:
        #     # logger.save_images(test_images, conditioned_labels, lookup_labels, epoch, full_image,partial_2fold,pixel)
        if epoch == num_epochs - 1 and pixel_to_label:
            print('training finished')
            # utils.save_generation_distribution(ex, epoch, categorization, generator, pixel_to_label)
            save_generation_distribution(ex, epoch, categorization, cat_names, generator, pixel_to_label,
                                         dataset['num_condition'], full_image, pixel, Quarter_fill, zdim,
                                         plot_generated_distribution=True)
            # logger.display_status(epoch, num_epochs, n_batch, 256, d_error, g_error, var_error, d_pred_real,
            #                       d_pred_fake, var_pred_real, var_pred_fake, gprediction_d, gprediction_var)
            break
        # logger.log_metrics(ex, class_accuracies, class_accuracies_quad, class_accuracies_without_unknowns,
        #                    class_accuracies_without_unknowns_quad, epoch, num_unknowns)
