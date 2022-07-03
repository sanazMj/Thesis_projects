
from torch.autograd.variable import Variable
import torch
import pickle
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

def create_test_samples(categorization, num_condition, pixel, zdim, test_samples_per_category=1000):
    """
        Creates labels for test samples"""
    num_test_samples = test_samples_per_category * num_condition
    # zdim = 100
    test_samples = noise(num_test_samples, zdim, pixel)

    labels_of_test_samples = Variable(torch.zeros(num_test_samples, categorization)).cuda()
    for i in range(num_test_samples):
        cat_idx = i % categorization
        labels_of_test_samples[i][cat_idx] = 1
    return test_samples, labels_of_test_samples

def convert_tuple_binary_to_int(input_tuple):
    temp = ''
    for i in range(len(input_tuple)):
        temp += str(int(input_tuple[i]))
    return temp
