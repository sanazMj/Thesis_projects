
import pandas as pd
import numpy as np
import torch
from sklearn.utils import shuffle
import pickle
def symmetric(image, full_image):
    # if not full_image:
    # image = image.reshape(int(np.sqrt(len(image))),int(np.sqrt(len(image))))
    for i in range(len(image)):
        for j in range(len(image)):
            if i != j:
                if image[i, j] != image[j, i]:
                    return False

    return True


def quad_from_partial(image):
    quad = []
    # len_img = int(np.sqrt(len(image)))
    # image = image.reshape(len_img,len_img)
    for i in range(len(image) - 1, -1, -1):
        for j in range(i, len(image)):
            quad.append(image[i, j])
    return np.array(quad)


def get_partial_three_fold_from_quad(quad_array, partial_len):
    partial = np.zeros((partial_len, partial_len))
    index = 0
    for i in range(partial_len - 1, -1, -1):
        for j in range(i, partial_len):
            partial[i, j] = quad_array[index]
            partial[j, i] = partial[i, j]  # Symmetry

            index += 1
    return partial

def Prepadding_images(images, complete_image,  padding_width):

    if complete_image:
        for i in range(len(images)):
            a = np.concatenate((images[i], images[i], images[i]))
            b = np.concatenate((a, a, a), axis=1)
            # Width  = complete_pixels.shape[1]
            # c = b[Width-pad_size : 2*Width + pad_size, Width-pad_size : 2*Width + pad_size]
            len_pixel = images[i].shape[0]
            c = b[len_pixel - padding_width:(2 * len_pixel) + padding_width,
                len_pixel - padding_width:(2 * len_pixel) + padding_width]
            c = np.reshape(c, (1, c.shape[0], c.shape[1]))
            if i == 0:
                extended = c
            else:
                extended = np.concatenate((extended, c), axis=0)
        Prepadded_images = extended.reshape((extended.shape[0], -1)).astype(int)
    else:
        for i in range(len(images)):
            a = np.concatenate((images[i], images[i]))
            b = np.concatenate((a, a), axis=1)
            c = b[4 - padding_width:9 + padding_width, 4 - padding_width:9 + padding_width]
            c = np.reshape(c, (1, c.shape[0], c.shape[1]))
            if i == 0:
                extended = c
            else:
                extended = np.concatenate((extended, c), axis=0)
        Prepadded_images = extended.reshape((extended.shape[0], -1)).astype(int)

    return Prepadded_images


def Prepadding_images_cuda(images, complete_image,  padding_width):

    if complete_image:
        for i in range(len(images)):
            a = torch.cat((images[i][0], images[i][0], images[i][0]))
            b = torch.cat((a, a, a),1)
            # Width  = complete_pixels.shape[1]
            # c = b[Width-pad_size : 2*Width + pad_size, Width-pad_size : 2*Width + pad_size]
            len_pixel = images[i][0].shape[0]
            c = b[len_pixel - padding_width:(2 * len_pixel) + padding_width,
                len_pixel - padding_width:(2 * len_pixel) + padding_width]
            c = c.view(1, c.shape[0], c.shape[1])
            if i == 0:
                extended = c
            else:
                extended = torch.cat((extended, c), 0)
        Prepadded_images = extended.view(extended.shape[0],extended.shape[1]*extended.shape[2])
    else:
        for i in range(len(images)):
            a = torch.cat((images[i][0], images[i][0]))
            b = torch.cat((a, a),1)
            c = b[4 - padding_width:9 + padding_width, 4 - padding_width:9 + padding_width]
            c = c.view(1, c.shape[0], c.shape[1])
            if i == 0:
                extended = c
            else:
                extended = torch.cat((extended, c),0)
        Prepadded_images = extended.view(extended.shape[0],extended.shape[1]*extended.shape[2])

    return Prepadded_images