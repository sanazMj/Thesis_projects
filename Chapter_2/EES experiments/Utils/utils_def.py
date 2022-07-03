from torch.autograd.variable import Variable
import torch
import pickle
import numpy as np
import random
import torch.nn.functional as f
# import tensorflow as tf

def pickle_load(dir):
    """Loads an object from given directory using pickle"""
    with open(dir, 'rb') as handle:
        obj = pickle.load(handle)
    return obj


def non_zero_dict(class_dict_list):
    len_non_zero = 0
    print(class_dict_list)
    for i in range(len(class_dict_list)):
        if class_dict_list[i] != 0:
            len_non_zero += 1
    return len_non_zero


def noise(size, zdim, pixel=False):
    # if pixel:
    #     n = Variable(torch.randn(size, pixel*pixel))
    # else:
    n = Variable(torch.randn(size, zdim))
    if torch.cuda.is_available(): return n.cuda()
    return n


def real_data_target(size):
    '''
    Tensor containing ones, with shape = size
    '''
    data = Variable(torch.ones(size, 1))
    if torch.cuda.is_available(): return data.cuda()
    return data


def fake_data_target(size):
    '''
    Tensor containing zeros, with shape = size
    '''
    data = Variable(torch.zeros(size, 1))
    if torch.cuda.is_available(): return data.cuda()
    return data

def make_label(size , value, gen_n):
    data = Variable(torch.zeros(size, gen_n+1))
    data.index_fill_(1, torch.tensor([value]) , 1)
    # print(data)
    return data.cuda()


# def compute_diversity_loss_tf(phi_fake, phi_real):
#     phi_fake = phi_fake.cpu().detach().numpy()
#     phi_fake = tf.convert_to_tensor(phi_fake)
#
#     phi_real = phi_real.cpu().detach().numpy()
#     phi_real = tf.convert_to_tensor(phi_real)
#
#     # phi_fake = tf.Tensor(phi_fake, dtype=tf.float64)
#     # phi_real = tf.Tensor(phi_real, dtype=tf.float64)
#
#     def compute_diversity_tf(phi):
#         phi = tf.nn.l2_normalize(phi, 1)
#         Ly = tf.tensordot(phi, tf.transpose(phi), 1)
#         eig_val, eig_vec = tf.linalg.eigh(Ly)
#         return eig_val, eig_vec
#
#     def normalize_min_max_tf(eig_val):
#         return tf.math.divide(tf.math.subtract(eig_val, tf.math.reduce_min(eig_val)),
#                       tf.math.subtract(tf.math.reduce_max(eig_val), tf.math.reduce_min(eig_val)))  # Min-max-Normalize Eig-Values
#
#     fake_eig_val, fake_eig_vec = compute_diversity_tf(phi_fake)
#     # print('Done')
#     real_eig_val, real_eig_vec = compute_diversity_tf(phi_real)
#     # print('Done')
#
#     # Used a weighing factor to make the two losses operating in comparable ranges.
#     eigen_values_loss = 0.0001 * tf.compat.v1.losses.mean_squared_error(labels=real_eig_val, predictions=fake_eig_val)
#     # print('Done')
#     eigen_vectors_loss = -tf.math.reduce_sum(tf.multiply(fake_eig_vec, real_eig_vec), 0)
#     # print('Done')
#     normalized_real_eig_val = normalize_min_max_tf(real_eig_val)
#     # print('Done')
#     weighted_eigen_vectors_loss = tf.math.reduce_sum(tf.math.multiply(normalized_real_eig_val, eigen_vectors_loss))
#     # print('Done reduce_sum')
#     # print(eigen_values_loss)
#     result = tf.cast(eigen_values_loss, tf.float32) + weighted_eigen_vectors_loss
#     # print('result', result)
#
#     # result = torch.tensor(result)
#     # print('result', result)
#
#     return result

def compute_diversity_loss(phi_fake, phi_real):
    def compute_diversity(phi):
        phi = f.normalize(phi, p=2, dim=1)
        S_B = torch.mm(phi, phi.t())
        # eig_vals, eig_vecs = torch.eig(S_B, eigenvectors=True)
        eig_vals, eig_vecs = torch.linalg.eig(S_B)
        # return eig_vals[:, 0], eig_vecs
        return eig_vals, eig_vecs

    def normalize_min_max(eig_vals):
        min_v, max_v = torch.min(eig_vals), torch.max(eig_vals)
        return (eig_vals - min_v) / (max_v - min_v)

    fake_eig_vals, fake_eig_vecs = compute_diversity(phi_fake)
    real_eig_vals, real_eig_vecs = compute_diversity(phi_real)
    # Scaling factor to make the two losses operating in comparable ranges.
    magnitude_loss = 0.0001 * f.mse_loss(target=real_eig_vals, input=fake_eig_vals)
    structure_loss = -torch.sum(torch.mul(fake_eig_vecs, real_eig_vecs), 0)
    normalized_real_eig_vals = normalize_min_max(real_eig_vals)
    weighted_structure_loss = torch.sum(torch.mul(normalized_real_eig_vals, structure_loss))
    return magnitude_loss + weighted_structure_loss

def create_test_samples(categorization, num_condition, pixel, zdim, test_samples_per_category=1000):
    """
        Creates labels for test samples"""
    num_test_samples = test_samples_per_category * num_condition
    # zdim = 100
    test_samples = noise(num_test_samples, zdim, pixel)

    labels_of_test_samples = Variable(torch.zeros(num_test_samples, num_condition)).cuda()
    for i in range(num_test_samples):
        cat_idx = i % num_condition
        labels_of_test_samples[i][cat_idx] = 1
    return test_samples, labels_of_test_samples


def convert_tuple_binary_to_int(input_tuple):
    temp = ''
    for i in range(len(input_tuple)):
        temp += str(int(input_tuple[i]))
    return temp


def oct2array(octList, even_flag=False, side=None):
    """converts list of octant values to square array,
    size is determined by length of the octant values and the even_flag
    """
    if side is None:
        side = parDim2side(len(octList), even_flag=even_flag)
    c1 = side // 2
    c2 = int((side + 1) // 2)

    # mask=np.concatenate(([np.concatenate((np.zeros(side-1-k),np.ones(k+1))) for k in range(c2)],np.zeros((c1,r))))
    return np.concatenate(([np.concatenate(
        (np.zeros(side - 1 - k), octList[(k * (k + 1)) // 2:((k + 2) * (k + 1)) // 2])) for k in range(c2)],
                           np.zeros((c1, side))))


def eightfold_sym2(array_in):
    """produces array with 8 fold symmetry based on the contents of the first octant of  array_in
    Added support for even length sides
    """
    array = array_in.copy()
    r = array.shape[0]
    c1 = r // 2
    c2 = (r + 1) // 2

    mask = np.concatenate(
        ([np.concatenate((np.zeros(r - 1 - k), np.ones(k + 1))) for k in range(c2)], np.zeros((c1, r))))

    marray = array * mask
    # print(marray)
    marray += np.rot90(np.rot90(np.transpose(marray)))
    for k in range(c2):
        marray[c2 - 1 - k, c1 + k] /= 2
    # print(marray)
    marray += np.fliplr(marray)
    if np.mod(r, 2) == 1:
        for k in range(c2):
            marray[k, c2 - 1] /= 2
        # print(marray)
    marray += np.flipud(marray)
    if np.mod(r, 2) == 1:
        for k in range(r):
            marray[c2 - 1, k] /= 2
    return marray


def create_sim_data(imgs, pixel, batch_size, Labels_classify, n_mixture, type=0, Level_line=0.5, Coef=1, slope=5):
    '''
    Construct the similar images for VARNET training
    :param imgs: generator output
    :return: The first element of generator ouput is repeated and passed
    '''
    if type == 0:
        num_var = [1, 2, 4, 8, 16, 32, 64, 128]
    elif type == 1:
        num_var = [1, 2, 4, 8, 16, 32, 64]
    elif type == 3:
        num_var = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]

        # num_var = [1, 2, 4, 8, 16, 32, 64]

    num = num_var[np.random.randint(0, len(num_var) - 1, 1)[0]]
    indexes = random.sample(list(range(imgs.shape[0])), num)

    # indexes = np.random.randint(0,imgs.shape[0]-1, num)
    fake_data = imgs[indexes, :]
    fake_label = Labels_classify[indexes, :]
    fake_data = fake_data.repeat(imgs.shape[0] // len(indexes), 1)
    fake_label = fake_label.repeat(Labels_classify.shape[0] // len(indexes), 1)
    # print(Labels_classify.shape,fake_label.shape )
    # if type == 0 :
    if num == 1:
        label = torch.tensor([0.0])
    else:
        if type == 3:
            label = torch.tensor(Level_line / (1 + Coef * np.exp(-(num / n_mixture) * slope))).type(torch.float)
        else:
            label = torch.tensor([0.51 - (1. / num)])

    fake_label_new = np.stack([np.tile(fake_label[i].cpu(), (pixel, pixel, 1)) for i in range(fake_label.shape[0])])
    # print(fake_label_new.shape)
    fake_label = torch.tensor(fake_label_new)
    # print(fake_label.shape )

    return fake_data, fake_label, label


class ModelSwitcher:
    """
    Select models based on properties
    """
    def __init__(self, args):
        [self.channels, self.Channel_factor, self.Kernel_factor, self.zdim, self.dataset_num_features, self.dataset_num_condition,
         self.dataset_num_pixels, self.Minibatch, self.minibatch_kind, self.Pack_number, self.PacGAN_pacnum, self.ndf, self.ngf,
         self.minibatch_net, self.mode_collapse, self.gen_num] = args

    def condition_to_model(self, pixel_full, partial_2fold, model_type, model_structure):
        method_name = "_".join([str(partial_2fold),  str(pixel_full),model_type,model_structure])
        print(method_name)
        method = getattr(self, method_name, lambda: "Undefined Model")
        return method()

    def initializer(self, initialize_models_func):
        generator, discriminator, varmeter, g_optimizer, d_optimizer, varmeter_optimizer, loss = initialize_models_func(
            self.channels, self.Channel_factor, self.Kernel_factor, self.zdim, self.dataset_num_condition,
            self.dataset_num_pixels, self.Minibatch, self.minibatch_kind, self.Pack_number, self.PacGAN_pacnum, self.mode_collapse, self.gen_num)

        return generator, discriminator, varmeter, g_optimizer, d_optimizer, varmeter_optimizer, loss

    def True_9_Vanilla_Convolutional(self):
        from EES.Main.Models.Partial_image.len_9.Model_9_Partial_CGAN_kernel3 import initialize_models
        return self.initializer(initialize_models)

    def True_9_Conditional_Convolutional(self):
        from EES.Main.Models.Partial_image.len_9.Model_9_Partial_cCGAN_kernel3 import initialize_models
        return self.initializer(initialize_models)

    def True_9_Vanilla_ConvOriginal(self):
        from EES.Main.Models.Partial_image.len_9.Model_9_CGAN_Original import initialize_models
        return self.initializer(initialize_models)

    def True_9_Conditional_ConvOriginal(self):
        from EES.Main.Models.Partial_image.len_9.Model_9_cCGAN_Original import initialize_models
        return self.initializer(initialize_models)

    def True_19_Vanilla_Convolutional(self):
        from EES.Main.Models.Partial_image.len_19.Model_19_partial_type1_CGAN import initialize_models
        return self.initializer(initialize_models)

    def True_19_Conditional_Convolutional(self):
        from EES.Main.Models.Partial_image.len_19.Model_19_Partial_Type1 import initialize_models
        return self.initializer(initialize_models)

    def True_19_Vanilla_ConvOriginal(self):
        from EES.Main.Models.Partial_image.len_19.Model_19_Partial_CGAN_Original import initialize_models
        return self.initializer(initialize_models)

    def True_19_Conditional_ConvOriginal(self):
        from EES.Main.Models.Partial_image.len_19.Model_19_Partial_cCGAN_Original import initialize_models
        return self.initializer(initialize_models)
    # elif Model_type == 'Vanilla' and Model_structure == 'FF' and num_hidden_layers == 2:
    #     from Models.Model_FF_Vanilla import initialize_models
    #     # from train import train_discriminator, train_generator, train_varmeter
    # elif Model_type == 'Conditional' and Model_structure == 'FF' and num_hidden_layers == 2:
    #     from Models.Model_FF_Conditional import initialize_models

    # if (Pixel_Full == 19 or Pixel_Full == 9) and (
    #         model_structure == 'Convolutional' or model_structure == 'ConvOriginal'):
    #     generator, discriminator, varmeter, g_optimizer, d_optimizer, varmeter_optimizer, loss = initialize_models(
    #         channels, Channel_factor, Kernel_factor, zdim, dataset_num_condition,
    #         dataset_num_pixels, Minibatch, minibatch_kind, Pack_number, PacGAN_pacnum)
    #     loss_dict = {}
    #     loss_dict['BCE'] = loss
    # elif model_structure == 'FF':
    #     generator, discriminator, varmeter, g_optimizer, d_optimizer, varmeter_optimizer, loss = initialize_models(
    #         dataset_num_features, dataset_num_condition, dataset_num_pixels, ndf, ngf, zdim, minibatch_net,
    #         Pack_number, PacGAN_pacnum)
    #     loss_dict = {}
    #     loss_dict['BCE'] = loss
    # else:
    #     generator, discriminator, g_optimizer, d_optimizer, loss_dict = initialize_models(Model, model_structure,
    #                                                                                       model_type, full_image,
    #                                                                                       Pixel_Full, Losses, args)
