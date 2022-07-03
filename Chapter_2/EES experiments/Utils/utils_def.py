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
   
