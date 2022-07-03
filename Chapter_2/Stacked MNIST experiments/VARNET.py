
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

    Model = 'VARNET' # [VARNET]
    Model_structure = 'FF' #['FF','Convolutional', 'ConvOriginal']
    Model_type = 'Vanilla' #['Conditional' 'Vanilla']
    mode_collapse = '' #'DSGAN, ' 'Minibatch', PacGAN, Minibatch
    model_kind = 'pacGAN' # pacGAN
    num_epochs = 1
    pac_dim = 1
    img_size = 32
    # var_pac = 4
    var_coef = 1


    batch_size = 60
    var_pac = batch_size

    num_training_sample = 100000
    # num_training_sample = 128000
    num_test_sample = 26000
    num_runs = 10
    n_features = 100
    discr_traing_rolls = 1
    test_noise_dim = 2000
    type_same = 2
    same = False
    num_condition = 10
    dir = '/home/sanaz/Ryerson/Projects/VARNETProject/Results_MNIST/'
@ex.automain
def main(Model, Model_structure, Model_type, model_kind, mode_collapse, img_size, num_epochs, batch_size, num_training_sample, num_test_sample,
          num_runs, type_same,same , num_condition, n_features, discr_traing_rolls, pac_dim, var_pac, var_coef, dir,test_noise_dim):

    torch.cuda.set_device(0)
    print(torch.cuda.is_available())
    id = ex.current_run._id
    if model_kind == 'pacGAN' and Model_structure != 'FF' :
        img_size = 64

    if Model_structure == 'FF' and Model_type == 'Vanilla':
        generator, discriminator, varnet, G_optimizer, D_optimizer, v_optimizer, loss, classifier = initialize_models_varnet(img_size, num_condition, pac_dim, var_pac, n_features=100)
    else:
        generator, discriminator, varnet, G_optimizer, D_optimizer, v_optimizer, loss, classifier = initialize_modelsnet_varnet(
            model_kind, img_size, var_pac, pac_dim, n_features=100)


    train_loader_Stacked, prob =  load_data(batch_size, img_size)

    # train_loader_Stacked, prob = load_data(num_training_sample, batch_size, img_size)
    # print(prob)
    # generator, discriminator, varnet, G_optimizer, D_optimizer, v_optimizer, loss, classifier = initialize_modelsnet_varnet(model_kind,img_size,  var_pac, pac_dim, n_features=100)

    train_hist = {}
    train_hist['D_losses'] = []
    train_hist['G_losses'] = []
    train_hist['V_losses'] = []
    train_hist['per_epoch_ptimes'] = []
    train_hist['total_ptime'] = []

    test_noise = torch.randn((test_noise_dim, 100))
    test_noise = Variable(test_noise.cuda())

    print('training start!')
    start_time = time.time()
    for epoch in range(num_epochs):
        print('epoch', epoch)
        D_losses = []
        G_losses = []
        V_losses = []
        # learning rate decay
        if (epoch + 1) == 11:
            G_optimizer.param_groups[0]['lr'] /= 10
            D_optimizer.param_groups[0]['lr'] /= 10
            print("learning rate change!")

        if (epoch + 1) == 16:
            G_optimizer.param_groups[0]['lr'] /= 10
            D_optimizer.param_groups[0]['lr'] /= 10
            print("learning rate change!")

        epoch_start_time = time.time()
        y_real_ = torch.ones(batch_size//pac_dim)
        y_fake_ = torch.zeros(batch_size//pac_dim)
        y_real_, y_fake_ = Variable(y_real_.cuda()), Variable(y_fake_.cuda())

        for x_, y_ in train_loader_Stacked:
            # print('batches',  x_, y_)
            # train discriminator D
            discriminator.zero_grad()

            mini_batch = x_.size()[0]

            if mini_batch != batch_size:
                break
                # y_real_ = torch.ones(mini_batch//pac_dim)
                # y_fake_ = torch.zeros(mini_batch//pac_dim)
                # y_real_, y_fake_ = Variable(y_real_.cuda()), Variable(y_fake_.cuda())

            # y_fill_ = fill[y_]
            x_ = Variable(x_.cuda())
            x_reshaped = x_.reshape(x_.shape[0] // pac_dim, x_.shape[1] * pac_dim,
                                x_.shape[2], x_.shape[3])
            # print(x_.shape)
            D_result = discriminator(x_reshaped).reshape(x_reshaped.shape[0])
            D_real_loss = loss(D_result, y_real_)

            z_ = torch.randn((mini_batch, 100))
            # y_ = (torch.rand(mini_batch, 1) * 10).type(torch.LongTensor).squeeze()
            # y_label_ = onehot[y_]
            # y_fill_ = fill[y_]
            z_ = Variable(z_.cuda())
            G_result = generator(z_)
            G_result_reshaped = G_result.reshape(G_result.shape[0] // pac_dim, G_result.shape[1] * pac_dim,
                                             G_result.shape[2], G_result.shape[3])
            # print(G_result.shape, G_result_reshaped.shape)
            D_result = discriminator(G_result_reshaped).reshape(G_result_reshaped.shape[0])
            D_fake_loss = loss(D_result, y_fake_)
            D_fake_score = D_result.data.mean()

            D_train_loss = D_real_loss + D_fake_loss

            D_train_loss.backward()
            D_optimizer.step()
            D_losses.append(D_train_loss.item())

            # print(D_train_loss.item())
            # D_losses.append(D_train_loss.data[0])
            ######################################################
            # train varnet V
            x_reshaped = x_.reshape(x_.shape[0] // var_pac, x_.shape[1] * var_pac, x_.shape[2], x_.shape[3])
            # For VARNET training
            if same==False:
                # print(x_.shape)
                fake_data_same, y_fake_reshaped = create_sim_data(x_,type_same)
                fake_data_same_reshape = fake_data_same.reshape(fake_data_same.shape[0] // var_pac,
                                                                fake_data_same.shape[1] * var_pac,  fake_data_same.shape[2], fake_data_same.shape[3])
                repeated_data = fake_data_same_reshape.cuda()
                # print(repeated_data.shape)
                y_fake_reshaped = y_fake_reshaped.repeat(repeated_data.shape[0])
                y_fake_reshaped = y_fake_reshaped.cuda()
                # print(y_fake_reshaped.shape)
            else:
                repeated_data = x_[0, :].reshape(1, x_.shape[1], x_.shape[2], x_.shape[3])
                repeated_data = repeated_data.repeat(x_.shape[0], 1, 1, 1)
                repeated_data = repeated_data.reshape(x_.shape[0] // var_pac, x_.shape[1] * var_pac, x_.shape[2], x_.shape[3])
                y_fake_reshaped = torch.zeros(mini_batch // var_pac)
                y_fake_reshaped =  Variable(y_fake_reshaped.cuda())
            y_real_reshaped = torch.ones(mini_batch // var_pac)
            y_real_reshaped  = Variable(y_real_reshaped.cuda())

            varnet.zero_grad()
            # print(varnet(x_reshaped).shape,y_real_reshaped.shape )
            v_result = varnet(x_reshaped).reshape(x_reshaped.shape[0])
            v_real_loss = loss(v_result, y_real_reshaped)

            # print(varnet(repeated_data).shape,y_fake_reshaped.shape )

            v_result = varnet(repeated_data).reshape(repeated_data.shape[0])
            v_same_loss = loss(v_result, y_fake_reshaped)

            loss_v = v_real_loss + v_same_loss
            # print(loss_v)
            loss_v.backward()
            v_optimizer.step()
            V_losses.append(loss_v.item())

            #############################################
            # train generator G
            generator.zero_grad()

            z_ = torch.randn((mini_batch, 100))
            # y_ = (torch.rand(mini_batch, 1) * 10).type(torch.LongTensor).squeeze()
            # y_label_ = onehot[y_]
            # y_fill_ = fill[y_]
            z_ = Variable(z_.cuda())

            G_result = generator(z_)
            G_result_reshaped_dim = G_result.reshape(G_result.shape[0] // pac_dim, G_result.shape[1] * pac_dim,
                                             G_result.shape[2], G_result.shape[3])
            G_result_reshaped = G_result.reshape(G_result.shape[0] // var_pac, G_result.shape[1] * var_pac,
                                                 G_result.shape[2], G_result.shape[3])

            D_result = discriminator(G_result_reshaped_dim).reshape(G_result_reshaped_dim.shape[0])
            G_train_loss = loss(D_result, y_real_)

            v_result = varnet(G_result_reshaped).reshape(G_result_reshaped.shape[0])
            v_real_loss = loss(v_result, y_real_reshaped) * var_coef

            (G_train_loss + v_real_loss).backward()
            G_optimizer.step()

            # G_losses.append(G_train_loss.data[0])
            G_losses.append((G_train_loss + v_real_loss).item())

        epoch_end_time = time.time()
        per_epoch_ptime = epoch_end_time - epoch_start_time

        print('[%d/%d] - ptime: %.2f, loss_d: %.3f, loss_g: %.3f, loss_v: %.3f' % (
        (epoch + 1), num_epochs, per_epoch_ptime, torch.mean(torch.FloatTensor(D_losses)),
        torch.mean(torch.FloatTensor(G_losses)), torch.mean(torch.FloatTensor(V_losses))))
        evaluation(100, 2000, generator, classifier, img_size, prob)
        print('test each  epoch on 2000')

        # train_hist['D_losses'].append(torch.mean(torch.FloatTensor(D_losses)))
        # train_hist['G_losses'].append(torch.mean(torch.FloatTensor(G_losses)))
        # train_hist['V_losses'].append(torch.mean(torch.FloatTensor(V_losses)))
        #
        train_hist['per_epoch_ptimes'].append(per_epoch_ptime)
        #
        if epoch >1 and epoch % 5 == 0:
            img = generator(test_noise).cpu().detach()
            img_transpose = torch.transpose(img, 1, 3)
            img_final = torch.tensor(merge(img_transpose[:100], [10, 10]))
            plt.imshow(img_final, interpolation='nearest')
            # plt.imshow(img_transpose[0], interpolation='nearest')
            plt.savefig(dir + '/Images/id_' + str(id) +'_'+str(epoch) + '.png')
            plt.show()
            print('test every 5 epochs on ', num_test_sample)

            evaluation(100, num_test_sample, generator, classifier, img_size, prob)

    img = generator(test_noise).cpu().detach()
    img_transpose = torch.transpose(img, 1, 3)
    img_final = torch.tensor(merge(img_transpose[:100], [10, 10]))
    plt.imshow(img_final, interpolation='nearest')
    # plt.imshow(img_transpose[0], interpolation='nearest')
    plt.savefig(dir + '/Images/id_' + str(id) + '.png')
    plt.show()

    end_time = time.time()
    total_ptime = end_time - start_time
    train_hist['total_ptime'].append(total_ptime)

    print("Avg one epoch ptime: %.2f, total %d epochs ptime: %.2f" % (
    torch.mean(torch.FloatTensor(train_hist['per_epoch_ptimes'])), num_epochs, total_ptime))
    print("Training finish!... save training results")
    print('test at the end epochs on ', num_test_sample)

    evaluation(100, num_test_sample, generator, classifier, img_size, prob)


