from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
from sacred import Experiment
from sacred.observers import FileStorageObserver

from utils import *
from Synthetic_data.models import *

ex = Experiment('test')
ex.observers.append(FileStorageObserver.create('logs'))

@ex.config
def my_config():

    Model = 'Normal' # [VARNET]
    Model_structure = 'FF' #['FF','Convolutional', 'ConvOriginal']
    Model_type = 'Vanilla' #['Conditional' 'Vanilla']
    mode_collapse = '' #'DSGAN, ' 'Minibatch', PacGAN, Minibatch
    dataset = 'grid' # ring, grid
    n_mixture = 25
    model_kind = 'normal' # pacGAN
    num_epochs = 1
    pac_dim = 1
    batch_size = 100
    training_samples = 100000
    test_samples = 26000
    # training_samples = 1000
    # test_samples = 260
    num_runs = 10
    n_features = 128
    discr_traing_rolls = 1
    test_noise_dim = 2000
    minibatch_net = False
    dir = '/home/sanaz/Ryerson/Projects/VARNETProject/Results/'
@ex.automain
def main(Model, Model_structure, Model_type, model_kind, mode_collapse, dataset, num_epochs, batch_size, training_samples, test_samples,
          num_runs, n_mixture, n_features, discr_traing_rolls, test_noise_dim, pac_dim, dir, minibatch_net):

    torch.cuda.set_device(0)
    print(torch.cuda.is_available())
    id = ex.current_run._id

    if dataset == 'ring':
        n_mixture = 8
        std_dev = 0.01
        trainset, tensor_loc, label = sample_gen_ring2D(training_samples, n_mixture, std_dev)
        trainset = trainset.type(torch.float32)
    elif dataset == 'grid':
        # n_mixture = 25
        # n_mixture = 36
        std_dev = 0.05#0.05
        trainset, tensor_loc, label = sample_gen_Grid2D(training_samples, n_mixture, std_dev)
        trainset = trainset.type(torch.float32)

    labels_training, min_dist_training, labels_without_thresh_training = classify(trainset, tensor_loc, label, std_dev)
    q, q_num_mode, q_without_thresh, q_num_mode_without_thresh = create_prob(labels_training,
                                                                             labels_without_thresh_training, n_mixture)

    # train_data = torch.utils.data.TensorDataset(trainset, labels_training,labels_without_thresh_training)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

    generator, discriminator, g_optim, d_optim, criterion = initialize_models(model_kind, minibatch_net, n_features=128, n_in=2, n_out=2, pac_dim=pac_dim)
    generator.train()
    discriminator.train()

    g_losses = []
    d_losses = []
    test_noise = noise(2000)
    for epoch in range(num_epochs):
        # print('epoch', epoch)
        g_error = 0.0
        d_error = 0.0
        for i, data in enumerate(trainloader):
            imgs = data
            n = len(imgs)
            for j in range(discr_traing_rolls):
                fake_data = generator(noise(n)).detach()
                real_data = imgs.cuda()
                real_data_reshaped = real_data.reshape(real_data.shape[0]//pac_dim, real_data.shape[1] * pac_dim)
                fake_data_reshaped = fake_data.reshape(fake_data.shape[0]//pac_dim ,fake_data.shape[1] * pac_dim)

                d_error_batch, prediction_real, prediction_fake = train_discriminator(d_optim, discriminator, criterion, real_data_reshaped, fake_data_reshaped)
                d_error += d_error_batch

            fake_data = generator(noise(n))
            fake_data_reshaped = fake_data.reshape(fake_data.shape[0] // pac_dim, fake_data.shape[1] * pac_dim)

            g_error_batch, pred_fake = train_generator(g_optim, discriminator, criterion, fake_data_reshaped)
            g_error += g_error_batch


        # num_mode, KL, JSD_score, KL_nonthresh, JSD_score_nonthresh, high_quality_samples = evaluate_modes_reverse_KL(
        #     generator, 1000, q, q_without_thresh, n_mixture, tensor_loc, label, std_dev=std_dev)

        g_losses.append(g_error / (i))
        d_losses.append(d_error / (i))
        print('Epoch {}: g_loss: {:.8f} d_loss: {:.8f}\r'.format(epoch, g_error / (i+1), d_error / (i+1)))
        print('Epoch {}: D(x): {:.8f} D(G(x)): {:.8f}\r'.format(epoch, prediction_real.mean(), prediction_fake.mean()))
        num_mode, KL, JSD_score, KL_nonthresh, JSD_score_nonthresh, high_quality_samples = evaluate_modes_reverse_KL(
            generator, test_noise_dim, q, q_without_thresh, n_mixture, tensor_loc, label, std_dev=std_dev)

        if epoch>1 and epoch % 50 == 0:
            print(epoch, '%50')
            img = generator(test_noise).cpu().detach()
            plt.figure()
            plt.scatter(trainset[:, 0], trainset[:, 1], c='b', edgecolor='none')
            plt.scatter(img[:, 0], img[:, 1], c='g', edgecolor='none')
            plt.savefig(dir + '/Images/id_' + str(id) + '_epoch_' + str(epoch) + '.png')
            # torch.save(generator.state_dict(), dir + '/Models/generator_' + str(id)+'.pt')

           
            # Evaluate
            num_mode, KL, JSD_score, KL_nonthresh, JSD_score_nonthresh, high_quality_samples = evaluate_modes_reverse_KL(
                generator, test_samples, q, q_without_thresh, n_mixture, tensor_loc, label, std_dev=std_dev)

    img = generator(test_noise).cpu().detach()
    plt.figure()
    plt.scatter(trainset[:, 0], trainset[:, 1], c='b', edgecolor='none')
    plt.scatter(img[:, 0], img[:, 1], c='g', edgecolor='none')
    plt.savefig(dir + '/Images/id_' + str(id)+'.png')

    print('Training Finished')
    torch.save(generator.state_dict(), dir + '/Models/generator_' + str(id)+'.pt')


    # imgs = [np.array(to_image(i)) for i in images]
    # imageio.mimsave('progress.gif', imgs)
    plt.figure()
    plt.plot(g_losses, label='Generator_Losses')
    plt.plot(d_losses, label='Discriminator Losses')
    plt.legend()
    plt.savefig(dir + '/Loss_images/Id_' + str(id) + '.png')
    # plt.savefig('loss.png')

    # Evaluate

    num_mode, KL, JSD_score, KL_nonthresh, JSD_score_nonthresh, high_quality_samples = evaluate_modes_reverse_KL(generator, test_samples, q, q_without_thresh, n_mixture, tensor_loc, label,std_dev=std_dev)

