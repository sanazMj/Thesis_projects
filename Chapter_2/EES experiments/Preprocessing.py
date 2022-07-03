import pandas as pd
import numpy as np
# import utils1
import torch
from sklearn.utils import shuffle
import pickle
from Utils.utils_image_edit import *
from Utils.utils_def import *
from Utils.population_generator_utils import *
from Utils.utils_log import *
def read_ees_dataset(categorization, batch_size, full_image, partial_2fold, Pixel_Full, Quarter_fill, prepadding,
                            prepadding_width, lookup=False, dir='Synthetic_data_binary/', balance=False):
    """
        @full_image: if true returns full image as features
        else returns 1/8th of the image, forces symmetry

        @lookup: If True, it means we have some of the search space.
        We create a mapping between pixels and labels

        @categorization specifies how to create the labels
        2: high pass low pass


    """
    dataset = {}
    cat_names = None
    """the indexes in filter_indexes array will be removed from the dataset but will be
        available in the lookup dictionary"""
    cat_labels, filter_indexes = [], []
    dir = '/home/sanaz/Ryerson/Projects/Master_code/Synthetic_data_binary/'

    # For every type of categorization create labels, and one_hot_labels variables
    if categorization == 2: # 1 or [0,1] is low pass, 0 or [1, 0] is high pass
        file = open(dir + 'data_partial9.pkl', 'rb')
        pixels = pickle.load(file)
        labels = np.array(list(pixels.values()))
        labels = torch.from_numpy(labels).type(torch.FloatTensor)

        L_max = Pixel_Full
        L_min = (Pixel_Full-1)//2

    elif categorization == 8:  # kmeans 8 cats
        # pixels = pd.read_csv(dir + 'pixels.txt', header=None, sep=' ').values
        Main = '/home/sanaz/Ryerson/Projects/GAN_Main_Project/Synthetic_data_binary/'

        with open(Main + "data_partial" + str(9) + ".pkl", 'rb') as f:
            pixels = pickle.load(f)
        pixels = np.array(list(pixels.keys()), dtype=np.uint8)
        # preprocessed file generated using exploration/k_means.ipnyb

        preds_dict = pickle_load(Main + '6_8_4categories/9x9_8_cats_kmeans_preds.pickle')
        one_hot_labels = preds_dict['preds']  # uint8 dtype
        labels = torch.from_numpy(one_hot_labels).type(torch.FloatTensor)
        cat_names = preds_dict['cat_names']  # cat names in string, Low Pass, High Pass etc.
        L_max = Pixel_Full
        L_min = (Pixel_Full-1)//2


    elif categorization == 2.1 or categorization == 2.2 or categorization == 2.3:
        bit_num = int((Pixel_Full//2 + 1)*(Pixel_Full//2+2)//2)
        data = np.load(dir + 'DatabaseLPandHP_Side'+str(Pixel_Full)+'_Bits'+str(bit_num)+'_TotalCount100000_LPcount50.0_HPcount50.0.npz') # 39x39
        # data = np.load(dir +'preprocessed_data/DatabaseLPandHP_Side39_Bits210_TotalCount1000000_LPcount50.0_HPcount50.0.npz') #19x19
        pixels = np.array(data['Data'], dtype=np.uint8)
        TRANSFER_FUNCTIONS = data['Labels']
        pixels, TRANSFER_FUNCTIONS = shuffle(pixels, TRANSFER_FUNCTIONS, random_state=0)

        if categorization == 2.2: # 100.000 samples
            pixels = pixels[:100000]
            TRANSFER_FUNCTIONS = TRANSFER_FUNCTIONS[:100000]
        elif categorization == 2.3: # 10.000 samples
            pixels = pixels[:10000]
            TRANSFER_FUNCTIONS = TRANSFER_FUNCTIONS[:10000]

        cat_names = [0, 1]
        one_hot_label_names = pd.get_dummies(cat_names).values

        cat_to_label, label_to_cat = {}, {}
        for cat, one_hot_label in zip(cat_names, one_hot_label_names):
            cat_to_label[cat] = tuple(one_hot_label)

        cat_labels = list(TRANSFER_FUNCTIONS)
        one_hot_labels = np.array([cat_to_label[cat] for cat in list(TRANSFER_FUNCTIONS)])
        L_max = Pixel_Full
        L_min = (Pixel_Full - 1) // 2

        one_hot_labels_filtered = np.array(np.delete(one_hot_labels, filter_indexes, axis=0).tolist())
        one_hot_labels_filtered = one_hot_labels_filtered.astype('uint8')
        labels = torch.from_numpy(one_hot_labels_filtered).type(torch.FloatTensor)

    else:
        raise ValueError('Error: categorization value')
    # if filter_indexes:




    if full_image: # Convert 1/8 to full image
        if categorization == 2:
            list_pixel_keys = list(pixels.keys())
            complete_pixels = np.stack(
                [eightfold_sym2(oct2array(list_pixel_keys[i])) for i in range(len(pixels))])
            if prepadding:

                # complete_pixels_list = complete_pixels.reshape((complete_pixels.shape[0], -1)).astype(int)

                complete_pixels = Prepadding_images(complete_pixels, True, prepadding_width)
                dataset['complete_pixels'] = torch.Tensor(complete_pixels)
                data = torch.utils.data.TensorDataset(dataset['complete_pixels'], labels)

            else:
                complete_pixels = complete_pixels.reshape((complete_pixels.shape[0], -1)).astype(int)
                dataset['complete_pixels'] = torch.Tensor(complete_pixels)
                data = torch.utils.data.TensorDataset(dataset['complete_pixels'], labels)
        else:
            complete_pixels = np.stack(
                [eightfold_sym2(oct2array(pixels[i])) for i in range(len(pixels))])
            if prepadding:

                # complete_pixels_list = complete_pixels.reshape((complete_pixels.shape[0], -1)).astype(int)

                complete_pixels = Prepadding_images(complete_pixels, True, prepadding_width)
                dataset['complete_pixels'] = torch.Tensor(complete_pixels)
                data = torch.utils.data.TensorDataset(dataset['complete_pixels'], labels)

            else:
                complete_pixels = complete_pixels.reshape((complete_pixels.shape[0], -1)).astype(int)
                dataset['complete_pixels'] = torch.Tensor(complete_pixels)
                data = torch.utils.data.TensorDataset(dataset['complete_pixels'], labels)

    else:

        if partial_2fold:

            if categorization == 2:
                list_pixel_keys = list(pixels.keys())
                complete_pixels = np.stack(
                    [eightfold_sym2(oct2array(list_pixel_keys[i])) for i in range(len(pixels))])
                if prepadding:
                    fold_2pixels_list = np.stack([complete_pixels[i, L_min:L_max, L_min:L_max] for i in range(complete_pixels.shape[0])])
                    fold_2pixels_list = fold_2pixels_list.reshape((fold_2pixels_list.shape[0], -1)).astype(int)
                    fold_2pixels = Prepadding_images(complete_pixels, False, prepadding_width)

                    dataset['fold_2pixels'] = torch.Tensor(fold_2pixels)
                    data = torch.utils.data.TensorDataset(dataset['fold_2pixels'], labels)

                else:
                    fold_2pixels = np.stack(
                        [complete_pixels[i, L_min:L_max, L_min:L_max] for i in range(complete_pixels.shape[0])])
                    if Quarter_fill:
                        fold_2pixels_list = []
                        for i in range(fold_2pixels.shape[0]):
                            for j in range(L_max - L_min):
                                for k in range(j):
                                    fold_2pixels[i][j][k] = 3
                            fold_2pixels_list.append(quad_from_partial(fold_2pixels[i]))
                    fold_2pixels = fold_2pixels.reshape((fold_2pixels.shape[0], -1)).astype(int)
                    dataset['fold_2pixels'] = torch.Tensor(fold_2pixels)
                    data = torch.utils.data.TensorDataset(dataset['fold_2pixels'], labels)

            else:
                complete_pixels = np.stack(
                    [eightfold_sym2(oct2array(pixels[i])) for i in range(len(pixels))])
                if prepadding:
                    fold_2pixels_list = np.stack(
                        [complete_pixels[i, L_min:L_max, L_min:L_max] for i in range(complete_pixels.shape[0])])
                    fold_2pixels_list = fold_2pixels_list.reshape((fold_2pixels_list.shape[0], -1)).astype(int)
                    fold_2pixels = Prepadding_images(complete_pixels, False, prepadding_width)

                    dataset['fold_2pixels'] = torch.Tensor(fold_2pixels)
                    data = torch.utils.data.TensorDataset(dataset['fold_2pixels'], labels)

                else:
                    fold_2pixels = np.stack(
                        [complete_pixels[i, L_min:L_max, L_min:L_max] for i in range(complete_pixels.shape[0])])
                    if Quarter_fill:
                        fold_2pixels_list = []
                        for i in range(fold_2pixels.shape[0]):
                            for j in range(L_max - L_min):
                                for k in range(j):
                                    fold_2pixels[i][j][k] = 3
                            fold_2pixels_list.append(quad_from_partial(fold_2pixels[i]))
                    fold_2pixels = fold_2pixels.reshape((fold_2pixels.shape[0], -1)).astype(int)
                    dataset['fold_2pixels'] = torch.Tensor(fold_2pixels)
                    data = torch.utils.data.TensorDataset(dataset['fold_2pixels'], labels)





    data_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True)
    dataset['num_batches'] = len(data_loader)
    dataset['num_condition'] = np.unique(labels, axis=0).shape[0]
    dataset['num_pixels'] = data[0][0].size()[0]
    dataset['num_features'] = dataset['num_condition'] + dataset['num_pixels']

    if lookup:
        if full_image:
            pixel_list = complete_pixels
        else:
            if partial_2fold:
                if prepadding:
                    pixel_list = fold_2pixels_list
                else:
                    if Quarter_fill:
                        pixel_list = fold_2pixels_list
                    else:
                        pixel_list = fold_2pixels

        label_list = labels
        pixel_to_label = {tuple(k):v for k, v in zip(pixel_list, label_list)}
    else:
        pixel_to_label = None

    return data_loader, dataset, pixel_to_label, cat_names


