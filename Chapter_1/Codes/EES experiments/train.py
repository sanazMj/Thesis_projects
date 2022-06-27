from Utils.utils_def import real_data_target, fake_data_target
import torch
import numpy as np
def train_discriminator(discriminator, loss_dict, optimizer, real_data, fake_data,Disc_Labels):
    loss = loss_dict['BCE']
    # Reset gradients
    optimizer.zero_grad()

    # 1.1 Train on Real Data
    prediction_real = discriminator(real_data, Disc_Labels)
    # Calculate error and backpropagate
    error_real = loss(prediction_real, real_data_target(real_data.size(0)))

    # 1.2 Train on Fake Data
    prediction_fake = discriminator(fake_data, Disc_Labels)
    # Calculate error and backpropagate
    error_fake = loss(prediction_fake, fake_data_target(real_data.size(0)))
    error = (error_fake + error_real)

    error.backward()
    # 1.3 Update weights with gradients
    optimizer.step()

    # Return error
    return error, prediction_real, prediction_fake


def train_generator(discriminator, loss_dict, optimizer, fake_data, z_noise, Disc_Labels, mode_collapse, iteration=0):
    loss = loss_dict['BCE']
    if len(loss_dict.values())>1:
        loss_sym = loss_dict['SYM']
    else:
        loss_sym = False

    # 2. Train Generator
    # Reset gradients
    optimizer.zero_grad()
    # Sample noise and generate fake data
    prediction = discriminator(fake_data, Disc_Labels)
    # Calculate error and backpropagate
    error_BCE = loss(prediction, real_data_target(prediction.size(0)))
    G_noise = 0
    if mode_collapse == 'DSGAN':
        dist_measure = 'rgb'
        noise_w = 5 * np.exp(iteration/100)
        # noise_w = 5  # diversity encouraging term
        B = int(fake_data.size(0) / 2)
        noise = z_noise
        # criterionFeat = torch.nn.L1Loss().cuda()
        # feat_weights = 4.0 / (opt.n_layers_D + 1)
        # D_weights = 1.0 / 1
        # for i in range(1):
        #     for j in range(len(D_result[i]) - 1):
        #         g_feat += D_weights * feat_weights * \
        #                   criterionFeat(D_result[i][j], pred_real[i][j].detach()) * opt.feat_w

        # noise sensitivity loss
        if dist_measure == 'rgb':
            g_noise_out_dist = torch.mean(torch.abs(fake_data[:B] - fake_data[B:]))
        # elif dist_measure == 'perceptual':
        #     g_noise_out_dist = 0
        #     for i in range(opt.num_D):
        #         for j in range(len(pred_fake[i]) - 1):
        #             g_noise_out_dist += D_weights * feat_weights * \
        #                                 torch.mean(torch.abs(pred_fake[i][j][:B] - pred_fake[i][j][B:]).view(B, -1),
        #                                            dim=1)

        g_noise_z_dist = torch.mean(torch.abs(noise[:B] - noise[B:]).view(B, -1), dim=1)
        # G_noise = torch.exp(-torch.mean(g_noise_out_dist / g_noise_z_dist) * noise_w)
        G_noise = torch.mean(g_noise_out_dist / g_noise_z_dist) * noise_w

    # print('g',G_noise)
    if loss_sym:
        error_sym = loss_sym(fake_data)
    else:
        error_sym = 0
    error = error_sym + error_BCE - G_noise
    # error = error_sym + error_BCE + G_noise
    error.backward()
    # Update weights with gradients
    optimizer.step()
    # Return error
    return error