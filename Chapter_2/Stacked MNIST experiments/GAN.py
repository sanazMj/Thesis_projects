
import time
import matplotlib.pyplot as plt
from sacred import Experiment
from sacred.observers import FileStorageObserver

from Stacked_MNIST.utils import *
from Stacked_MNIST.models import *

ex = Experiment('test')
ex.observers.append(FileStorageObserver.create('logs'))

@ex.config
def my_config():

    Model = 'normal' # [VARNET]
    Model_structure = 'FF' #['FF','Convolutional', 'ConvOriginal']
    Model_type = 'Vanilla' #['Conditional' 'Vanilla']
    mode_collapse = '' #'DSGAN, ' 'Minibatch', PacGAN, Minibatch
    model_kind = 'pacGAN' # pacGAN
    num_epochs = 1
    pac_dim = 1
    img_size = 32
    var_pac = 4
    var_coef = 1
    batch_size = 100
    num_training_sample = 100000
    # num_training_sample = 1000
    # num_test_sample = 260
    num_condition = 10
    # num_training_sample = 128000
    num_test_sample = 26000
    num_runs = 10
    n_features = 100
    discr_traing_rolls = 1
    # test_noise_dim = 2000
    dir = '/home/sanaz/Ryerson/Projects/VARNETProject/Results_MNIST/'
@ex.automain
def main(Model, Model_structure, Model_type, model_kind, num_condition, mode_collapse, img_size, num_epochs, batch_size, num_training_sample, num_test_sample,
          num_runs, n_features, discr_traing_rolls, pac_dim, var_pac, var_coef, dir):
    print(pac_dim)
    torch.cuda.set_device(1)
    print(torch.cuda.is_available())
    if model_kind == 'pacGAN':
        img_size = 64
    id = ex.current_run._id
    if Model_structure == 'FF' and  Model_type == 'Vanilla':
        generator, discriminator, G_optimizer, D_optimizer, loss, classifier = initialize_models(img_size, num_condition, pac_dim, n_features=100)
    else:
        generator, discriminator, G_optimizer, D_optimizer, loss, classifier = initialize_models_net(model_kind,
                                                                                                     img_size,
                                                                                                     n_features,
                                                                                                     pac_dim)
    print('model loaded')


    train_loader_Stacked, prob =  load_data(batch_size, img_size)
    print('data loaded')

    # train_loader_Stacked, prob = load_data(num_training_sample, batch_size, img_size)
    # print(prob)

    # generator, discriminator, G_optimizer, D_optimizer, loss, classifier = initialize_models(model_kind, n_features, pac_dim)

    train_hist = {}
    train_hist['D_losses'] = []
    train_hist['G_losses'] = []
    train_hist['per_epoch_ptimes'] = []
    train_hist['total_ptime'] = []

    test_noise = torch.randn((2000, 100))
    test_noise = Variable(test_noise.cuda())
    print('training start!')
    start_time = time.time()
    for epoch in range(num_epochs):
        print('epoch', epoch)
        D_losses = []
        G_losses = []

        # learning rate decay
        # if (epoch + 1) == 11:
        #     G_optimizer.param_groups[0]['lr'] /= 10
        #     D_optimizer.param_groups[0]['lr'] /= 10
        #     print("learning rate change!")
        #
        # if (epoch + 1) == 16:
        #     G_optimizer.param_groups[0]['lr'] /= 10
        #     D_optimizer.param_groups[0]['lr'] /= 10
        #     print("learning rate change!")


        epoch_start_time = time.time()
        y_real_ = torch.ones(batch_size//pac_dim)
        y_fake_ = torch.zeros(batch_size//pac_dim)
        y_real_, y_fake_ = Variable(y_real_.cuda()), Variable(y_fake_.cuda())

        for num_batches, (x_, y_) in enumerate(train_loader_Stacked):
            # print('batches',  num_batches)

            discriminator.zero_grad()

            mini_batch = x_.size()[0]

            if mini_batch != batch_size:
                y_real_ = torch.ones(mini_batch//pac_dim)
                y_fake_ = torch.zeros(mini_batch//pac_dim)
                y_real_, y_fake_ = Variable(y_real_.cuda()), Variable(y_fake_.cuda())

            # y_fill_ = fill[y_]
            x_ = Variable(x_.cuda())
            # print(x_.shape)
            x_reshaped = x_.reshape(x_ .shape[0] // pac_dim, x_.shape[1] * pac_dim,
                                                 x_.shape[2], x_.shape[3])
            D_result = discriminator(x_reshaped).squeeze()
            D_real_loss = loss(D_result, y_real_)

            z_ = torch.randn((mini_batch, 100))
            # y_ = (torch.rand(mini_batch, 1) * 10).type(torch.LongTensor).squeeze()
            # y_label_ = onehot[y_]
            # y_fill_ = fill[y_]
            z_ = Variable(z_.cuda())
            G_result = generator(z_)
            # print(G_result.shape, z_.shape)
            G_result_reshaped = G_result.reshape(G_result.shape[0] // pac_dim, G_result.shape[1] * pac_dim,
                                                 G_result.shape[2], G_result.shape[3])
            # print(G_result_reshaped.shape)
            D_result = discriminator(G_result_reshaped).squeeze()

            D_fake_loss = loss(D_result, y_fake_)
            D_fake_score = D_result.data.mean()

            D_train_loss = D_real_loss + D_fake_loss

            D_train_loss.backward()
            D_optimizer.step()
            D_losses.append(D_train_loss.item())

            #############################################
            # train generator G
            generator.zero_grad()

            z_ = torch.randn((mini_batch, 100))
            z_ = Variable(z_.cuda())

            G_result = generator(z_)
            G_result_reshaped = G_result.reshape(G_result.shape[0] // pac_dim, G_result.shape[1] * pac_dim,
                                                 G_result.shape[2], G_result.shape[3])
            # print(G_result_reshaped.shape)
            D_result = discriminator(G_result_reshaped).squeeze()
            G_train_loss = loss(D_result, y_real_)

            G_train_loss.backward()
            G_optimizer.step()

            # G_losses.append(G_train_loss.data[0])
            G_losses.append((G_train_loss).item())

        epoch_end_time = time.time()
        per_epoch_ptime = epoch_end_time - epoch_start_time

        print('[%d/%d] - ptime: %.2f, loss_d: %.3f, loss_g: %.3f' % (
        (epoch + 1), num_epochs, per_epoch_ptime, torch.mean(torch.FloatTensor(D_losses)),
        torch.mean(torch.FloatTensor(G_losses))))
        print('test every epoch on 2000')
        evaluation(100, 2000, generator, classifier, img_size, prob)
        # if epoch >1 and epoch % 10 == 0:

        if epoch >1 and epoch % 5 == 0:
            img = generator(test_noise).cpu().detach()
            img_transpose = torch.transpose(img, 1, 3)
            img_final = torch.tensor(merge(img_transpose[:100], [10,10]))
            plt.imshow(img_final, interpolation='nearest')
            plt.savefig(dir + '/Images/id_' + str(id) +'_'+str(epoch) + '.png')
            plt.show()
            print('test every 5 epochs on ',num_test_sample)

            # evaluation(2000, num_test_sample, generator, classifier, img_size, prob)
            evaluation(100, num_test_sample, generator, classifier, img_size, prob)

        # train_hist['D_losses'].append(torch.mean(torch.FloatTensor(D_losses)))
        # train_hist['G_losses'].append(torch.mean(torch.FloatTensor(G_losses)))
        # train_hist['V_losses'].append(torch.mean(torch.FloatTensor(V_losses)))
        #
        train_hist['per_epoch_ptimes'].append(per_epoch_ptime)
        #
    end_time = time.time()
    total_ptime = end_time - start_time
    train_hist['total_ptime'].append(total_ptime)

    img = generator(test_noise).cpu().detach()
    # img_transpose = torch.transpose(img, 1, 3)
    # plt.imshow(img_transpose[0], interpolation='nearest')
    img_transpose = torch.transpose(img, 1, 3)
    img_final = torch.tensor(merge(img_transpose[:100], [10, 10]))
    plt.imshow(img_final, interpolation='nearest')
    plt.savefig(dir + '/Images/id_' + str(id) + '.png')
    plt.show()

    print("Avg one epoch ptime: %.2f, total %d epochs ptime: %.2f" % (
    torch.mean(torch.FloatTensor(train_hist['per_epoch_ptimes'])), num_epochs, total_ptime))
    print("Training finish!... save training results")
    print('test at the end epochs on ', num_test_sample)

    evaluation(100, num_test_sample, generator, classifier, img_size, prob)


