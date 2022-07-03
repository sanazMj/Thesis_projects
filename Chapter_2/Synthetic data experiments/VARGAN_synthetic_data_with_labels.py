from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
from sacred import Experiment
from sacred.observers import FileStorageObserver

from Synthetic_data.utils_with_labels import *
from Synthetic_data.models import *

ex = Experiment('test')
ex.observers.append(FileStorageObserver.create('logs'))

@ex.config
def my_config():

    Model = 'VARNET' # [VARNET]
    Model_structure = 'FF' #['FF','Convolutional', 'ConvOriginal']
    Model_type = 'Vanilla' #['Conditional' 'Vanilla']
    mode_collapse = '' #'DSGAN, ' 'Minibatch', PacGAN, Minibatch
    dataset = 'grid' # ring, grid
    model_kind = 'normal' # pacGAN
    var_coef = 1 # the portion  of varnet loss effective in generator
    pac_dim =1
    num_epochs = 1
    n_mixture = 36
    batch_size = n_mixture * 4 #100
    pac_var = batch_size #100 #4# The level of packing in varnet
    random_sample = 2 # If 0  np.random.randint, if 1 : random.sample if 2: label_selection function
    training_samples = 100000
    test_samples = 26000
    # training_samples = 1000
    # test_samples = 260
    num_runs = 1
    n_features = 128
    discr_traing_rolls = 1
    test_noise_dim = 2000
    same_creation_type = 1
    minibatch_net = False
    dir = '/home/Projects/VARNETProject/'
@ex.automain
def main(Model, Model_structure, Model_type, model_kind, mode_collapse, dataset, var_coef, pac_var, num_epochs, batch_size, training_samples, test_samples,
          num_runs,n_mixture, n_features, random_sample, discr_traing_rolls, test_noise_dim, pac_dim, dir, same_creation_type, minibatch_net):

    torch.cuda.set_device(0)
    print(torch.cuda.is_available())
    id = ex.current_run._id

    if dataset == 'ring':
        n_mixture = 8
        std_dev = 0.01
        trainset, tensor_loc, label = sample_gen_ring2D(training_samples, n_mixture, std_dev)
        trainset = trainset.type(torch.float32)
        if same_creation_type == 2:
            pac_var = [3, 4, 5]
            batch_size = get_lcms(3, 4, 5) * 3
    elif dataset == 'grid':
        std_dev = 0.05 #0.05
        trainset, tensor_loc, label = sample_gen_Grid2D(training_samples, n_mixture, std_dev)
        trainset = trainset.type(torch.float32)
        if same_creation_type == 2:
            if n_mixture == 25:
                pac_var = [4, 15, 20]
                batch_size = get_lcms(4, 15, 20) * 3

            elif n_mixture == 36:
                pac_var = [16, 18, 24]
                batch_size = get_lcms(16, 18, 24) * 3
    print('pac',pac_var, batch_size)
    labels_training, min_dist_training, labels_without_thresh_training = classify(trainset, tensor_loc, label, std_dev)
    q, q_num_mode, q_without_thresh, q_num_mode_without_thresh = create_prob(labels_training, labels_without_thresh_training, n_mixture)

    labels_without_thresh_training = labels_without_thresh_training.reshape(labels_without_thresh_training.shape[0],1)
    labels_without_thresh_training = torch.tensor(labels_without_thresh_training)

    train_data = torch.cat((trainset, labels_without_thresh_training), axis=1)
    trainloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    generator, discriminator, varnet, g_optim, d_optim, v_optim, criterion = initialize_models_varnet(model_kind,minibatch_net, pac_var, same_creation_type, pac_dim, n_features=128, n_in=2, n_out=2)
    generator.train()
    discriminator.train()
    num_var = 5
    g_losses = []
    d_losses = []
    v_losses = []
    test_noise = noise(1000)
    for epoch in range(num_epochs):
        g_error = 0.0
        d_error = 0.0
        v_error = 0.0
        for i, data in enumerate(trainloader):

            imgs = data[:,:2]
            Label_images = data[:,2:]
            # print(Label_images.shape)
            n = len(imgs)
            if n < batch_size:
                break
            for j in range(discr_traing_rolls):
                fake_data = generator(noise(n)).detach()
                real_data = imgs.cuda()
                # print('diff data')

                real_data_reshape_list, total_label_diff = create_dif_data(real_data, pac_var, batch_size, type=same_creation_type)

                real_data_reshape_dim = real_data.reshape(real_data.shape[0] // pac_dim, real_data.shape[1] * pac_dim).cuda()
                fake_data_reshape_dim = fake_data.reshape(fake_data.shape[0] // pac_dim, fake_data.shape[1] * pac_dim).cuda()


                d_error_batch, pred_real, pred_fake = train_discriminator(d_optim, discriminator, criterion, real_data_reshape_dim, fake_data_reshape_dim)
                d_error += d_error_batch

             

                v_error_batch, pred_train, pred_same = train_varnet(v_optim, varnet, criterion, real_data_reshape_list,total_label_diff, real_data , num_var, pac_var, batch_size, n_mixture,Label_images, random_sample, same_creation_type)
                # print(pred_train, pred_same)
                v_error += v_error_batch

            fake_data = generator(noise(n))
            fake_data_dim = fake_data.reshape(fake_data.shape[0] // pac_dim, fake_data.shape[1] * pac_dim)
            # print('Gen data')

            fake_data1, label1  = create_dif_data(fake_data, pac_var, batch_size, type=same_creation_type)
             # fake_data1 = fake_data.reshape(fake_data.shape[0] // pac_var, fake_data.shape[1] * pac_var)

            var_coef_change = var_coef
            # print(var_coef_change)
            g_error_batch, pred_d, pred_v = train_generator_varnet(g_optim, discriminator, varnet, criterion, var_coef_change, pac_var, fake_data_dim, fake_data1,label1, same_creation_type=same_creation_type)
            g_error += g_error_batch

        g_losses.append(g_error / i)
        d_losses.append(d_error / i)
        v_losses.append(v_error / i)

        print('Epoch {}: g_loss: {:.8f} d_loss: {:.8f} v_loss: {:.8f}\r'.format(epoch, g_error / i, d_error / i, v_error / i))
        print('Epoch {}: D(x): {:.8f} D(G(x)): {:.8f}\r'.format(epoch, pred_real.mean(), pred_fake.mean()))
        print('Epoch {}: V(x): {:.8f} V(same): {:.8f}\r'.format(epoch, pred_train.mean(), pred_same.mean()))
        print('Epoch {}: D(G(x)): {:.8f} V(G(x)): {:.8f}\r'.format(epoch, pred_d.mean(), pred_v.mean()))
        num_mode, KL, JSD_score, KL_nonthresh, JSD_score_nonthresh, high_quality_samples = evaluate_modes_reverse_KL(
            generator, test_noise_dim, q, q_without_thresh, n_mixture, tensor_loc, label, std_dev=std_dev)

        if epoch>1 and epoch%50 == 0:
            print(epoch, '%50')

            img = generator(test_noise).cpu().detach()
            plt.figure()
            plt.scatter(trainset[:, 0], trainset[:, 1], c='b', edgecolor='none')
            plt.scatter(img[:, 0], img[:, 1], c='g', edgecolor='none')
            plt.savefig(dir + '/Images/id_' + str(id) +'_epoch_'+ str(epoch) + '.png')

            # Evaluate
            num_mode, KL, JSD_score, KL_nonthresh, JSD_score_nonthresh, high_quality_samples = evaluate_modes_reverse_KL(
                generator, test_samples, q, q_without_thresh, n_mixture, tensor_loc, label, std_dev=std_dev)

    # # GAN
    plt.figure()

    img = generator(test_noise).cpu().detach()
    plt.scatter(trainset[:, 0], trainset[:, 1], c='b', edgecolor='none')
    plt.scatter(img[:, 0], img[:, 1], c='g', edgecolor='none')
    plt.savefig(dir + '/Images/id_' + str(id)+'.png')
    print('Training Finished')
    # torch.save(generator.state_dict(), dir + '/Models/generator_' + str(id)+'.pt')


    # Evaluate
    num_mode, KL, JSD_score, KL_nonthresh, JSD_score_nonthresh, high_quality_samples = evaluate_modes_reverse_KL(generator, test_samples, q, q_without_thresh, n_mixture, tensor_loc, label,std_dev=std_dev)
