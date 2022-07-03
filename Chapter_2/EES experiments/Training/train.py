from EES.Main.Utils.utils_def import *
import numpy as np
import torch
from torch.autograd import Variable
import torch.autograd as autograd



def train_discriminator_MADGAN(optimizer, discriminator, criterion,
                               real_data, fake_data, j, gen_numbers):
    n = real_data.size(0)

    optimizer.zero_grad()

    prediction_real = discriminator(real_data)
    # print(prediction_real.shape,  make_label(n , gen_numbers-1, gen_numbers).shape)
    error_real = criterion(prediction_real, make_label(n , gen_numbers-1, gen_numbers))
    # print(error_real)
    error_real.backward()

    prediction_fake = discriminator(fake_data)
    error_fake = criterion(prediction_fake, make_label(n , j, gen_numbers))

    error_fake.backward()
    optimizer.step()

    return error_real + error_fake, prediction_real, prediction_fake



def train_generator_MADGAN(optimizer, discriminator, criterion, fake_data,
                           gen_numbers):
    n = fake_data.size(0)
    optimizer.zero_grad()

    prediction = discriminator(fake_data)
    error = criterion(prediction, make_label(n, gen_numbers-1, gen_numbers))

    error.backward()
    optimizer.step()

    return error, prediction





def train_discriminator_GDPP(discriminator, loss, optimizer, real_data,
                             fake_data, lables_list, lables_list_real):
    # Reset gradients

    optimizer.zero_grad()

    # 1.1 Train on Real Data
    prediction_real, last_layer_real = discriminator(real_data, lables_list_real)
    # Calculate error and backpropagate
    error_real = loss(prediction_real, real_data_target(real_data.size(0)))
    error_real.backward()

    # 1.2 Train on Fake Data
    prediction_fake, last_layer_fake = discriminator(fake_data, lables_list)
    # Calculate error and backpropagate
    error_fake = loss(prediction_fake, fake_data_target(real_data.size(0)))
    error_fake.backward()
    # D_loss = -torch.mean(torch.log(prediction_real) + torch.log(1 - prediction_fake))

    # gradient_penalty = compute_gradient_penalty(discriminator, real_data, fake_data,lables_list)
    # error_penalty = lambda_gp * gradient_penalty
    # error = -torch.mean(prediction_real) + torch.mean(prediction_fake) + error_penalty
    # error.backward()

    # 1.3 Update weights with gradients
    optimizer.step()

    # Return error
    return error_real + error_fake , prediction_real, prediction_fake
    # return error, prediction_real, prediction_fake


def train_generator_GDPP(discriminator, loss, optimizer, fake_data, real_data,
                         batch_labels, batch_real_labels):
    # 2. Train Generator
    # Reset gradients

    optimizer.zero_grad()
    # Sample noise and generate fake data
    prediction, last_layer_fake = discriminator(fake_data, batch_labels)
    # Calculate error and backpropagate
    error = loss(prediction, real_data_target(prediction.size(0)))


    prediction_real, last_pre_layer_real = discriminator(real_data, batch_real_labels)
    diversity_error = compute_diversity_loss(last_layer_fake, last_pre_layer_real)
    ##
    # diversity_error = compute_diversity_loss_tf(last_layer_fake, last_pre_layer_real)
    # diversity_error = diversity_error.cpu().numpy()
    # diversity_error = torch.tensor(diversity_error).cuda()
    ##
    error_var = 0.5 * (error + diversity_error)
    error_all = error_var
    error_all.backward()
    # Update weights with gradients
    optimizer.step()
    # Return error
    return error, prediction


def train_discriminator(discriminator, loss, optimizer, real_data, fake_data, lables_list):
    # Reset gradients

    optimizer.zero_grad()

    # 1.1 Train on Real Data
    prediction_real = discriminator(real_data, lables_list)
    # Calculate error and backpropagate
    error_real = loss(prediction_real, real_data_target(real_data.size(0)))
    error_real.backward()

    # 1.2 Train on Fake Data
    prediction_fake = discriminator(fake_data, lables_list)
    # Calculate error and backpropagate
    error_fake = loss(prediction_fake, fake_data_target(real_data.size(0)))
    error_fake.backward()
    # D_loss = -torch.mean(torch.log(prediction_real) + torch.log(1 - prediction_fake))

    # gradient_penalty = compute_gradient_penalty(discriminator, real_data, fake_data,lables_list)
    # error_penalty = lambda_gp * gradient_penalty
    # error = -torch.mean(prediction_real) + torch.mean(prediction_fake) + error_penalty
    # error.backward()

    # 1.3 Update weights with gradients
    optimizer.step()

    # Return error
    return error_real + error_fake , prediction_real, prediction_fake
    # return error, prediction_real, prediction_fake


def train_generator(discriminator, VarNET, loss, optimizer, fake_data, z_noise,  fake_data1,batch_labels, lables_list, mode_collapse, iterator=0):
    # 2. Train Generator
    # Reset gradients

    optimizer.zero_grad()
    # Sample noise and generate fake data
    prediction = discriminator(fake_data, batch_labels)
    # Calculate error and backpropagate
    error = loss(prediction, real_data_target(prediction.size(0)))

    prediction_var = VarNET(fake_data1, lables_list)
    error_var = loss(prediction_var, real_data_target(prediction_var.size(0))) * iterator

    G_noise = 0
    if mode_collapse == 'DSGAN':
        dist_measure = 'rgb'
        noise_w = 5 * np.exp(iterator/100)# diversity encouraging term
        B = int(fake_data.size(0) / 2)
        noise = z_noise

        # noise sensitivity loss
        if dist_measure == 'rgb':
            g_noise_out_dist = torch.mean(torch.abs(fake_data[:B] - fake_data[B:]))


        g_noise_z_dist = torch.mean(torch.abs(noise[:B] - noise[B:]).view(B, -1), dim=1)
        G_noise = torch.mean(g_noise_out_dist / g_noise_z_dist) * noise_w

    error_all = error  + error_var - G_noise
    error_all.backward()
    # Update weights with gradients
    optimizer.step()
    # Return error
    return error, prediction, prediction_var

def train_varmeter(varmeter, loss, optimizer,real_data,batch_VAR, batch_labels_reshape, batch_VAR_label, label_target):
    # Reset gradients
    optimizer.zero_grad()

    # 1.1 Train on Real Data
    prediction_real = varmeter(real_data, batch_labels_reshape)
    # Calculate error and backpropagate
    error_real = loss(prediction_real, real_data_target(real_data.size(0)))
    error_real.backward()

    # 1.2 Train on Fake Data
    # print(batch_VAR.shape, batch_VAR_label.shape)
    prediction_fake = varmeter(batch_VAR, batch_VAR_label)
    # Calculate error and backpropagate
    # error_fake = loss(prediction_fake, fake_data_target(real_data.size(0)))
    error_fake = loss(prediction_fake, label_target)

    error_fake.backward()

    optimizer.step()

    # Return error
    return error_real + error_fake , prediction_real, prediction_fake

