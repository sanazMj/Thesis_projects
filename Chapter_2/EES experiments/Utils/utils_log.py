import os
import numpy as np
import errno
import torchvision.utils as vutils
from tensorboardX import SummaryWriter
from IPython import display
from matplotlib import pyplot as plt
import torch
import logging
from sklearn.metrics import roc_curve, confusion_matrix, precision_recall_curve, precision_recall_fscore_support
import pickle
from collections import defaultdict
import time
from Utils.population_generator_utils import edges_connect2
from Utils.utils_def import *
from Utils.utils_image_edit import *
class Logger:

    def __init__(self, model_name, data_name, id, image_size, model_type, dir_summary=None):

        self.model_name = model_name
        self.data_name = data_name
        self.id = id
        if image_size:
            self.image_size = 'Full'
        else:
            self.image_size = 'Partial'

        self.model_type = model_type

        if dir_summary:
            self.comment = dir_summary
        else:
            self.comment = '{}_{}'.format(model_name, data_name)  # directory for summary
        self.data_subdir = '{}/{}'.format(model_name, data_name)  # subdirectory for data (images and models)

        self.writer = SummaryWriter(comment=self.comment)  # TensorBoard

        logging.basicConfig(filename='logs/logs_{}.log', level=logging.DEBUG)  # log messages for debugging
        self.logger = logging.getLogger(str(id))

    def log_message(self, message):
        self.logger.debug(message)

    def log(self, d_error, g_error, epoch, n_batch, num_batches,r_error =None):

        # var_class = torch.autograd.variable.Variable
        if isinstance(d_error, torch.autograd.Variable):
            d_error = d_error.data.cpu().numpy()
        if isinstance(g_error, torch.autograd.Variable):
            g_error = g_error.data.cpu().numpy()

        if isinstance(r_error, torch.autograd.Variable) and r_error:
            r_error = r_error.data.cpu().numpy()
        step = Logger._step(epoch, n_batch, num_batches)
        self.writer.add_scalar(
            '{}/D_error'.format(self.comment), d_error, step)

        self.writer.add_scalar(
            '{}/G_error'.format(self.comment), g_error, step)

        if r_error:
            self.writer.add_scalar(
                '{}/R_error'.format(self.comment), r_error, step)

    def log_optimizer(self, d_optimizer, g_optimizer, optimizer_name):
        """This function can be used to print dynamic learning rates or other optimizer parameters"""
        things_to_print = ['lr', 'momentum', 'weight_decay']
        for d_param_group, g_param_group in zip(d_optimizer.param_groups, g_optimizer.param_groups):
            for param in things_to_print:
                self.writer.add_scalar('{}/d_{}_{}'.format(self.comment, param, d_param_group[param]))
                self.writer.add_scalar('{}/g_{}_{}'.format(self.comment, param, d_param_group[param]))

    def log_images(self, images, num_images, epoch, n_batch, num_batches, format='NCHW', normalize=True):
        '''
        input images are expected in format (NCHW)
        '''
        if type(images) == np.ndarray:
            images = torch.from_numpy(images)

        if format == 'NHWC':
            images = images.transpose(1, 3)

        step = Logger._step(epoch, n_batch, num_batches)
        img_name = '{}/images{}'.format(self.comment, '')

        # Make horizontal grid from image tensor
        horizontal_grid = vutils.make_grid(
            images, normalize=normalize, scale_each=True)
        # Make vertical grid from image tensor
        nrows = int(np.sqrt(num_images))
        grid = vutils.make_grid(
            images, nrow=nrows, normalize=True, scale_each=True)

        # Add horizontal images to tensorboard
        self.writer.add_image(img_name, horizontal_grid, step)

        # Save plots
        self.save_torch_images(horizontal_grid, grid, epoch, n_batch)

    def log_parameters(self, generator, discriminator, epoch):
        for name, param in generator.named_parameters():
            self.writer.add_histogram('Generator.' + name, param.clone().cpu().data.numpy(), epoch)
        for name, param in discriminator.named_parameters():
            self.writer.add_histogram('Discriminator.' + name, param.clone().cpu().data.numpy(), epoch)

    def log_model_graph(self, model, dummy_input, model_name):
        """Saves graph of a model to tensorboard. Requires dummy input with right dimensions"""
        self.writer.add_graph(model, (dummy_input,))

    def log_figure(self, tag, figure, step):
        self.writer.add_figure(tag, figure, step)

    def save_images(self, test_images, conditioned_labels, lookup_labels, epoch, full_image, partial_2Fold, image_len):
        num_test_samples = conditioned_labels.shape[0]
        height, width = int(num_test_samples ** (1 / 2)), int(num_test_samples ** (1 / 2))
        fig, axrr = plt.subplots(height, width, figsize=(10, 10))
        plt.tight_layout()
        ims = []
        if full_image:
            ims = [test_image.reshape((image_len, image_len)) for test_image in test_images]
        else:
            if partial_2Fold:
                for test_image in test_images:
                    a = test_image.reshape((image_len, image_len))
                    full_image_len = 2 * image_len - 1
                    c = np.zeros((full_image_len, full_image_len))
                    c[image_len - 1:full_image_len, image_len - 1:full_image_len] = a
                    for i in range(image_len - 1, full_image_len):
                        for j in range(image_len - 1, full_image_len):
                            c[full_image_len - 1 - i, j] = c[i, j]
                            c[i, full_image_len - 1 - j] = c[i, j]
                            c[full_image_len - 1 - i, full_image_len - 1 - j] = c[i, j]
                    ims.append(c)
            else:
                ims = [eightfold_sym2(oct2array(test_image)).reshape((image_len, image_len)) for test_image in
                       test_images]

        for idx, (ax, im) in enumerate(zip(axrr.ravel(), ims)):
            ax.set_title('Conditioned:{}, True:{}'.format(conditioned_labels[idx], lookup_labels[idx]))
            if lookup_labels[idx] == conditioned_labels[idx]:
                ax.imshow(im, cmap='Greys', clim=[0, 1])
            else:
                ax.imshow(im, cmap='Reds', clim=[0, 1])
            ax.axis('off')

        out_dir = './results/{}_{}_{}/'.format(self.id, self.image_size, self.model_type)
        Logger._make_dir(out_dir)
        fig.savefig('{}/results_epoch_{}.png'.format(out_dir, epoch))
        self.log_figure('results', fig, epoch)

    def save_torch_images(self, horizontal_grid, grid, epoch, n_batch, plot_horizontal=True):
        out_dir = './data/images/{}'.format(self.data_subdir)
        Logger._make_dir(out_dir)

        # Plot and save horizontal
        fig = plt.figure(figsize=(16, 16))
        plt.imshow(np.moveaxis(horizontal_grid.numpy(), 0, -1))
        plt.axis('off')
        if plot_horizontal:
            display.display(plt.gcf())
        self._save_images(fig, epoch, n_batch, 'hori')
        plt.close()

        # Save squared
        fig = plt.figure()
        plt.imshow(np.moveaxis(grid.numpy(), 0, -1))
        plt.axis('off')
        self._save_images(fig, epoch, n_batch)
        plt.close()

    def _save_images(self, fig, epoch, n_batch, comment=''):
        out_dir = './data/images/{}'.format(self.data_subdir)
        Logger._make_dir(out_dir)
        fig.savefig('{}/{}_epoch_{}_batch_{}.png'.format(out_dir,
                                                         comment, epoch, n_batch))

    def display_status(self, epoch, num_epochs, n_batch, num_batches, d_error, g_error, var_error, d_pred_real,
                       d_pred_fake, var_pred_real, var_pred_fake, gprediction_d, gprediction_var):

        # var_class = torch.autograd.variable.Variable
        if isinstance(d_error, torch.autograd.Variable):
            d_error = d_error.data.cpu().numpy()
        if isinstance(g_error, torch.autograd.Variable):
            g_error = g_error.data.cpu().numpy()
        if isinstance(d_pred_real, torch.autograd.Variable):
            d_pred_real = d_pred_real.data
        if isinstance(d_pred_fake, torch.autograd.Variable):
            d_pred_fake = d_pred_fake.data
        if isinstance(var_error, torch.autograd.Variable):
            var_error = var_error.data.cpu().numpy()
        if isinstance(var_pred_real, torch.autograd.Variable):
            var_pred_real = var_pred_real.data
        if isinstance(var_pred_fake, torch.autograd.Variable):
            var_pred_fake = var_pred_fake.data

        print('Epoch: [{}/{}], Batch Num: [{}/{}]'.format(
            epoch, num_epochs, n_batch, num_batches)
        )
        print('Discriminator Loss: {:.4f}, Generator Loss: {:.4f}, , Var Loss: {:.4f}'.format(d_error, g_error,
                                                                                              var_error))
        print('D(x): {:.4f}, D(G(z)): {:.4f} , Var(x): {:.4f}, Var(same): {:.4f}, D(G): {:.4f}, V(G): {:.4f}'.format(
            d_pred_real.mean(), d_pred_fake.mean(), var_pred_real.mean(), var_pred_fake.mean(), gprediction_d.mean(),
            gprediction_var.mean()))

    def save_models(self, generator, discriminator, epoch):
        out_dir = './data/models/{}'.format(self.data_subdir)
        Logger._make_dir(out_dir)
        torch.save(generator.state_dict(),
                   '{}/G_epoch_{}'.format(out_dir, epoch))
        torch.save(discriminator.state_dict(),
                   '{}/D_epoch_{}'.format(out_dir, epoch))

    # def log_metrics(self, ex, class_accuracies, epoch, total_unknowns_Hp = None, total_unknowns_Lp = None):
    #     for name, accuracy in class_accuracies.items():
    #         self.writer.add_scalar('{}/{}'.format(self.comment, name), accuracy, epoch)
    #         ex.log_scalar(name, accuracy, epoch)
    #     if total_unknowns_Hp:
    #         self.writer.add_scalar('{}/{}'.format(self.comment, 'unknowns_HP'), total_unknowns_Hp, epoch)
    #         ex.log_scalar('unknowns_HP', total_unknowns_Hp, epoch)
    #     if total_unknowns_Lp:
    #         self.writer.add_scalar('{}/{}'.format(self.comment, 'unknowns_LP'), total_unknowns_Lp, epoch)
    #         ex.log_scalar('unknowns_LP', total_unknowns_Lp, epoch)
    def log_metrics(self, ex, class_accuracies, class_accuracies_quad, class_accuracies_without_unknowns,
                    class_accuracies_without_unknowns_quad, epoch, total_unknowns=None):
        print('Logging metrics')
        for name, accuracy in class_accuracies.items():
            self.writer.add_scalar('{}/{}'.format(self.comment, name), accuracy, epoch)
            ex.log_scalar(name, accuracy, epoch)

        for name, accuracy in class_accuracies_quad.items():
            self.writer.add_scalar('{}/{}'.format(self.comment, name + '_quad'), accuracy, epoch)
            ex.log_scalar(name + '_quad', accuracy, epoch)

        for name, accuracy in class_accuracies_without_unknowns.items():
            self.writer.add_scalar('{}/{}'.format(self.comment, name + '_accuracy_without_unknowns'), accuracy, epoch)
            ex.log_scalar(name + '_accuracy_without_unknowns', accuracy, epoch)

        for name, accuracy in class_accuracies_without_unknowns_quad.items():
            self.writer.add_scalar('{}/{}'.format(self.comment, name + '_accuracy_without_unknowns_quad'), accuracy,
                                   epoch)
            ex.log_scalar(name + '_accuracy_without_unknowns_quad', accuracy, epoch)

        if total_unknowns:
            self.writer.add_scalar('{}/{}'.format(self.comment, 'unknowns'), total_unknowns, epoch)
            ex.log_scalar('unknowns', total_unknowns, epoch)
        print('Logged metrics')

    def close(self):
        self.writer.close()

    # Private Functionality

    @staticmethod
    def _step(epoch, n_batch, num_batches):
        return epoch * num_batches + n_batch

    @staticmethod
    def _make_dir(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


def predict_test_samples(test_images, labels_of_test_noise, categorization, pixel_to_label, full_image, pixel,
                         Quarter_Fill):
    labels_of_test_noise = labels_of_test_noise[:test_images.shape[0]]  # Remove extra rows
    total_unknowns = 0
    if categorization in [2,8]:
        # # True labels found by lookup since we have all the search space
        lookup_labels = []
        lookup_labels_quad = []
        quad = False
        for index, image in enumerate(test_images):

            if Quarter_Fill:
                image = image[0]
                image = quad_from_partial(image)
                if tuple(image) in pixel_to_label:
                    lookup_labels.append(int(np.argmax(pixel_to_label[tuple(image)])))
                else:
                    lookup_labels.append('unknown')
            else:
                image = image.reshape((image.shape[0], -1)).astype(int)
                image = image[0]
                if tuple(image) in pixel_to_label:
                    lookup_labels.append(int(np.argmax(pixel_to_label[tuple(image)])))
                else:
                    lookup_labels.append('unknown')

            if quad:
                a = test_images[index][0]
                image = quad_from_partial(a)
                if tuple(image) in pixel_to_label:
                    lookup_labels_quad.append(int(np.argmax(pixel_to_label[tuple(image)])))
                else:
                    lookup_labels_quad.append('unknown')

        conditioned_labels = np.argmax(labels_of_test_noise.cpu().numpy(), axis=1)  # Conditioned on
        return conditioned_labels, lookup_labels, lookup_labels_quad, total_unknowns
    elif categorization == 2.1 or categorization == 2.2 or categorization == 2.3:
        lookup_labels = []
        lookup_labels_quad = []
        quad = True
        # assert test_images.shape[1] == len(list(pixel_to_label.keys())[0])
        for image in test_images:
            image = image.astype('uint8')
            image = image.reshape(image.shape[1], image.shape[2])
            if not full_image:
                if Quarter_Fill:
                    d = quad_from_partial(image)
                    full_pixel = 2 * pixel - 1
                    lpf_flag = int(edges_connect2(d, full_pixel))
                    lookup_labels.append(lpf_flag)

                else:
                    if symmetric(image, False):
                        d = quad_from_partial(image)
                        full_pixel = 2 * pixel - 1
                        lpf_flag = int(1 - edges_connect2(d, full_pixel))
                        lookup_labels.append(lpf_flag)  # If true it is low pass

                    else:
                        lookup_labels.append('unknown')
                    if quad:
                        d = quad_from_partial(image)
                        full_pixel = 2 * pixel - 1
                        lpf_flag = int(edges_connect2(d, full_pixel))
                        lookup_labels_quad.append(lpf_flag)
            else:
                # this should be corrected
                image = image.reshape(image.shape[0] * image.shape[1])
                full_pixel = 2 * pixel - 1
                lpf_flag = int(1 - edges_connect2(image, full_pixel))
                lookup_labels.append(lpf_flag)  # If true it is low pass

        conditioned_labels = np.argmax(labels_of_test_noise.cpu().numpy(), axis=1)
        return conditioned_labels, lookup_labels, lookup_labels_quad, 0

    else:
        raise ValueError('Unknown categorization in predict test samples function')


def create_accuracy_matrix(num_condition, conditioned_labels, lookup_labels, lookup_labels_quad, cat_names):
    """omit_unknowns: When True the unknown samples are omitted from the accuracy calculation"""
    accuracy_matrix = np.zeros((num_condition, 2))
    accuracy_matrix_without_unknowns = np.zeros((num_condition, 2))
    for conditioned_label, lookup_label in zip(conditioned_labels, lookup_labels):

        if lookup_label == 'unknown':
            accuracy_matrix[conditioned_label, 0] += 1  # this line added later, accuracy calculations before biweekly meeting 7 don't include this line
            continue

        if conditioned_label == lookup_label:
            accuracy_matrix[conditioned_label, 1] += 1
            accuracy_matrix_without_unknowns[conditioned_label, 1] += 1
        else:
            accuracy_matrix[conditioned_label, 0] += 1
            accuracy_matrix_without_unknowns[conditioned_label, 0] += 1

    accuracy_matrix_quad = np.zeros((num_condition, 2))
    accuracy_matrix_without_unknowns_quad = np.zeros((num_condition, 2))
    for conditioned_label, lookup_label in zip(conditioned_labels, lookup_labels_quad):
        if lookup_label == 'unknown':
            accuracy_matrix_quad[
                conditioned_label, 0] += 1  # this line added later, accuracy calculations before biweekly meeting 7 don't include this line
            continue

        if conditioned_label == lookup_label:
            accuracy_matrix_quad[conditioned_label, 1] += 1
            accuracy_matrix_without_unknowns_quad[conditioned_label, 1] += 1
        else:
            accuracy_matrix_quad[conditioned_label, 0] += 1
            accuracy_matrix_without_unknowns_quad[conditioned_label, 0] += 1

    class_accuracies = {}
    class_accuracies_without_unknowns = {}
    class_accuracies_quad = {}
    class_accuracies_without_unknowns_quad = {}

    if num_condition == 2:
        class_accuracies['low_pass'] = accuracy_matrix[1][1] / (accuracy_matrix[1][1] + accuracy_matrix[1][0])
        class_accuracies_without_unknowns['low_pass_without_unknowns'] = accuracy_matrix_without_unknowns[1][1] / (
                    accuracy_matrix_without_unknowns[1][1] + accuracy_matrix_without_unknowns[1][0])

        class_accuracies['high_pass'] = accuracy_matrix[0][1] / (accuracy_matrix[0][1] + accuracy_matrix[0][0])
        class_accuracies_without_unknowns['high_pass_without_unknowns'] = accuracy_matrix_without_unknowns[0][1] / (
                accuracy_matrix_without_unknowns[0][1] + accuracy_matrix_without_unknowns[0][0])

        class_accuracies_quad['low_pass'] = accuracy_matrix_quad[1][1] / (
                    accuracy_matrix_quad[1][1] + accuracy_matrix_quad[1][0])
        class_accuracies_without_unknowns_quad['low_pass_without_unknowns'] = accuracy_matrix_without_unknowns_quad[1][
                                                                                  1] / (
                                                                                      accuracy_matrix_without_unknowns_quad[
                                                                                          1][1] +
                                                                                      accuracy_matrix_without_unknowns_quad[
                                                                                          1][0])

        class_accuracies_quad['high_pass'] = accuracy_matrix_quad[0][1] / (
                    accuracy_matrix_quad[0][1] + accuracy_matrix_quad[0][0])
        class_accuracies_without_unknowns_quad['high_pass_without_unknowns'] = accuracy_matrix_without_unknowns_quad[0][
                                                                                   1] / (
                                                                                       accuracy_matrix_without_unknowns_quad[
                                                                                           0][1] +
                                                                                       accuracy_matrix_without_unknowns_quad[
                                                                                           0][0])
    else:

        for idx, cat in enumerate(cat_names):
            class_accuracies[str(cat)] = accuracy_matrix[idx][1] / (accuracy_matrix[idx][1] + accuracy_matrix[idx][0])
            class_accuracies_without_unknowns[str(cat)] = accuracy_matrix_without_unknowns[idx][1] / (
                        accuracy_matrix_without_unknowns[idx][1] + accuracy_matrix_without_unknowns[idx][0])

            class_accuracies_quad[str(cat)] = accuracy_matrix_quad[idx][1] / (
                        accuracy_matrix_quad[idx][1] + accuracy_matrix_quad[idx][0])
            class_accuracies_without_unknowns_quad[str(cat)] = accuracy_matrix_without_unknowns_quad[idx][1] / (
                    accuracy_matrix_without_unknowns_quad[idx][1] + accuracy_matrix_without_unknowns_quad[idx][0])

    return accuracy_matrix, class_accuracies, accuracy_matrix_quad, class_accuracies_quad, accuracy_matrix_without_unknowns, class_accuracies_without_unknowns, accuracy_matrix_without_unknowns_quad, class_accuracies_without_unknowns_quad


def check_for_mode_collapse(accuracy_matrix):
    created_classes = accuracy_matrix[:, 1]
    num_classes_created = np.sum(created_classes > 0)
    if num_classes_created < 2:
        print('Mode collapse observed, less than two classes are generated\n')
        print('Accuracy matrix: ', accuracy_matrix, '\n')


# def predict_test_samples(test_images, labels_of_test_noise, categorization, pixel_to_label):
#     total_unknowns_Hp = 0
#     total_unknowns_Lp = 0
#     conditioned_labels = np.argmax(labels_of_test_noise.cpu().numpy(), axis=1)  # Conditioned on
#     if categorization == 2:
#         # True labels found by lookup since we have all the search space
#         lookup_labels = []
#         for index, image in enumerate(test_images):
#             image = image.reshape((image.shape[0], -1)).astype(int)
#             image = image[0]
#             if tuple(image) in pixel_to_label:
#                 lookup_labels.append(np.argmax(pixel_to_label[tuple(image)]))
#             else:
#                 lookup_labels.append('unknown')
#                 if conditioned_labels[index] == 0:
#                     total_unknowns_Hp += 1
#                 else:
#                     total_unknowns_Lp += 1
#
#
#         return conditioned_labels, lookup_labels, total_unknowns_Hp,total_unknowns_Lp
#
# def create_accuracy_matrix(conditioned_labels, lookup_labels):
#     ## TODO Update function for n number of unique labels
#     conf_matrix = np.zeros((2,2))
#     for conditioned_label, lookup_label in zip(conditioned_labels, lookup_labels):
#         if lookup_label == 'unknown':
#             if conditioned_label == 0:
#                 conf_matrix[0,1] += 1
#             else:
#                 conf_matrix[1,0] += 1
#         elif conditioned_label == lookup_label:
#             conf_matrix[conditioned_label, lookup_label] += 1
#         elif conditioned_label==0 and lookup_label == 1:
#             # conf_matrix[1,0] += 1
#             conf_matrix[0, 1] += 1
#         elif conditioned_label==1 and lookup_label == 0:
#             # conf_matrix[0,1] += 1
#             conf_matrix[1, 0] += 1
#         else:
#             raise('Error creating conf matrix')
#     return conf_matrix

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


# Following two functions are for printing pixelated EES
# Ex: plt.imshow(eightfold_sym2(oct2array(octant_of_data)),cmap='Greys',clim=[0,1])
def oct2array(octList, even_flag=False, side=None):
    """converts list of octant values to square array,
    size is determined by length of the octant values and the even_flag
    """
    if side is None:
        side = parDim2side(len(octList), even_flag=even_flag)  # side is 9 for 15 pixels
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


def parDim2side(par_dim, even_flag=False):
    """given the length of the octant paramatrisation returns the size of the symmetric square array
    since there is an ambiguity in this mapping because some squares
    have the same size octant representation a flag for even of odd needs to be set
    """
    side = int(-2 + np.sqrt(1 + 8 * par_dim))
    if even_flag:
        side += 1
    return (side)


# def save_generation_distribution(ex, epoch, categorization, generator, pixel_to_label):
#     # num_test_samples = 300000
#     num_test_samples = 20000
#     if categorization == 2:
#         test_noise, labels_of_test_noise = create_test_samples(categorization, num_test_samples=num_test_samples)
#         test_images = generator(test_noise, labels_of_test_noise).data.cpu().numpy()
#         test_images = np.where(test_images > 0.5, 1, 0)  # convert to binary output
#
#         conditioned_labels, lookup_labels, total_unknowns_Hp,total_unknowns_Lp = predict_test_samples(
#             test_images,
#             labels_of_test_noise,
#             categorization,
#             pixel_to_label)
#
#         accuracy_matrix = create_accuracy_matrix(conditioned_labels, lookup_labels)
#         print('accuracy_matrix out of 20000', accuracy_matrix)
#
#         pixel_to_sum = {pixel:0 for pixel in pixel_to_label} # Count number of times each pixel appeared
#         for test_image in test_images:
#             test_image = test_image.reshape((test_image.shape[0], -1)).astype(int)
#             test_image = test_image[0]
#             if tuple(test_image) in pixel_to_sum:
#                 pixel_to_sum[tuple(test_image)] += 1
#
#         class_dict = {}
#         for pixel, label in pixel_to_label.items():
#             if tuple(label) not in class_dict:
#                 class_dict[tuple(label)] = []
#             class_dict[tuple(label)].append(pixel_to_sum[pixel])
#
#         print('Generated class distribution: ')
#         for aclass in class_dict:
#             print(aclass, ': ', len(class_dict[aclass]))
#
#         # key = list(class_dict.keys())[0]
#         # num_label = len(class_dict[key])
#         # plt.bar([i for i in range(0, num_label)], class_dict[key])
#         # plt.show()
#
#         start=0
#         my_colors = 'rgbkymcr'
#         for idx, aclass in enumerate(class_dict):
#             num_label = len(class_dict[aclass])
#             plt.bar([i for i in range(start, start+num_label)], class_dict[aclass], color=my_colors[idx])
#             start += num_label
#             # What percentage of possible configurations are generated
#             unique_generated = sum([1 for val in class_dict[aclass] if val > 0])
#             print('Configurations generated for class: ', aclass, ' is :',
#                   unique_generated, 'out of: ', num_label)
#             ex.log_scalar(aclass, unique_generated, epoch)
#         id = ex.current_run._id
#         plt.savefig('logs/{}/generated_class_distribution_{}_samples.png'.format(id, num_test_samples))
#         with open('logs/{}/class_distribution.pickle'.format(id), 'wb') as handle:
#             pickle.dump(class_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
#         return
#
#
#     else:
#         return
#
def save_generation_distribution(ex, epoch, categorization, cat_names, generator, pixel_to_label, num_condition,
                                 full_image, pixel_len, Quarter_Fill, zdim,
                                 plot_generated_distribution=False):
    start_time = time.time()
    # torch.cuda.empty_cache()
    quad = True
    # test_samples_per_category = 5000 # partil

    for i in range(10):
        test_samples_per_category = 2000
        num_test_samples = test_samples_per_category * num_condition
        test_noise, labels_of_test_noise = create_test_samples(categorization, num_condition, pixel_len,zdim,
                                                               test_samples_per_category=test_samples_per_category)

        test_images = generator(test_noise, labels_of_test_noise).data.cpu().numpy()
        test_images = np.where(test_images > 0.5, 1, 0)  # convert to binary output
        if i==0:
            test_images_all = test_images
            test_noise_all = test_noise
            test_noise_labels =labels_of_test_noise
        else:
            test_images_all = np.concatenate((test_images_all, test_images), 0)
            test_noise_all = torch.cat((test_noise_all, test_noise), 0)
            test_noise_labels = torch.cat((test_noise_labels, labels_of_test_noise), 0)

    test_images = test_images_all
    test_noise = test_noise_all
    labels_of_test_noise = test_noise_labels
    print('{} samples generated in {} seconds.'.format(test_images.shape, time.time() - start_time))
    test_samples_per_category = 20000
    num_test_samples = num_condition * test_samples_per_category
    conditioned_labels, lookup_labels, lookup_labels_quad, num_unknowns = predict_test_samples(
        test_images,
        labels_of_test_noise,
        categorization,
        pixel_to_label,
        full_image, pixel_len, Quarter_Fill)

    accuracy_matrix, class_accuracies, accuracy_matrix_quad, class_accuracies_quad, accuracy_matrix_without_unknowns, class_accuracies_without_unknowns, accuracy_matrix_without_unknowns_quad, class_accuracies_without_unknowns_quad = create_accuracy_matrix(
        num_condition, conditioned_labels,
        lookup_labels, lookup_labels_quad, cat_names)

    print('accuracy_matrix out of 40000', accuracy_matrix)
    print('accuracy_matrix out of 40000 quad', accuracy_matrix_quad)

    print('accuracy_matrix without unknowns out of 40000', accuracy_matrix_without_unknowns)
    print('accuracy_matrix without unknowns quad out of 40000', accuracy_matrix_without_unknowns_quad)

    uniqueValues, occurCount = np.unique(test_images, axis=0, return_counts=True)
    ex.log_scalar('Total uniquely generated samples', len(uniqueValues))
    pixel_to_label_new = {}
    pixel_to_label_new_quad = {}
    if True:
        pixel_to_sum = {pixel: 0 for pixel in pixel_to_label}  # Count number of times each pixel appeared
        pixel_to_sum_quad = {pixel: 0 for pixel in pixel_to_label}  # Count number of times each pixel appeared

        pixel_to_sum_convert = {convert_tuple_binary_to_int(pixel): 0 for pixel in pixel_to_label}  # Count number of times each pixel appeared
        pixel_to_sum_quad_convert = {convert_tuple_binary_to_int(pixel): 0 for pixel in pixel_to_label}
    for index,test_image in enumerate(test_images):
            if Quarter_Fill:
                test_image = quad_from_partial(test_image.reshape(pixel_len, pixel_len))
            else:
                test_image = test_image.reshape(pixel_len*pixel_len)
                if quad:
                    test_image1 = quad_from_partial(test_image.reshape(pixel_len, pixel_len))
                    test_image1 = get_partial_three_fold_from_quad(test_image1, pixel_len)
                    test_image1 = test_image1.reshape(pixel_len*pixel_len)
                    if tuple(test_image1) in pixel_to_sum_quad:
                        pixel_to_sum_quad[tuple(test_image1)] += 1
                        pixel_to_sum_quad_convert[convert_tuple_binary_to_int(test_image1)] +=1
                    else:
                        if lookup_labels_quad[index] != 'unknown':
                            if lookup_labels_quad[index] == 0:
                                pixel_to_label_new_quad[tuple(test_image1)] = torch.tensor([1, 0])
                            else:
                                pixel_to_label_new_quad[tuple(test_image1)] = torch.tensor([0, 1])
                            pixel_to_sum_quad[tuple(test_image1)] = 1  # New pixel
                            pixel_to_sum_quad_convert[convert_tuple_binary_to_int(test_image1)] = 1

            if tuple(test_image) in pixel_to_sum:
                pixel_to_sum[tuple(test_image)] += 1
                pixel_to_sum_convert[convert_tuple_binary_to_int(test_image)] += 1

            else:
                 if lookup_labels[index] !='unknown':
                    pixel_to_sum[tuple(test_image)] =1
                    pixel_to_sum_convert[convert_tuple_binary_to_int(test_image)] = 1

                    if lookup_labels[index] == 0:
                        pixel_to_label_new[tuple(test_image)] = torch.tensor([1, 0])
                    else:
                        pixel_to_label_new[tuple(test_image)] = torch.tensor([0, 1])






    class_dict = {}
    class_dict_quad = {}
    class_dict_pixels = {}
    class_dict_pixels_quad = {}
    # print('keys', pixel_to_sum_convert.keys())
    # print('values', pixel_to_sum_convert.values())
    #
    #
    # print('keys quad', pixel_to_sum_quad_convert.keys())
    # print('values quad', pixel_to_sum_quad_convert.values())
    # print('pixel', len(pixel_to_label), len(pixel_to_label_new), len(pixel_to_label_new_quad))
    for pixel, label in pixel_to_label.items():
        if tuple(np.array(label)) not in class_dict:
            class_dict[tuple(np.array(label))] = []
        if pixel_to_sum[pixel] != 0:
            class_dict[tuple(np.array(label))].append(pixel_to_sum[pixel])
    if len(pixel_to_label_new)>0:
        for pixel, label in pixel_to_label_new.items():
            if tuple(np.array(label)) not in class_dict:
                class_dict[tuple(np.array(label))] = []
            if pixel_to_sum[pixel] !=0:
                class_dict[tuple(np.array(label))].append(pixel_to_sum[pixel])
    if not Quarter_Fill:
        for pixel, label in pixel_to_label.items():
            if tuple(np.array(label)) not in class_dict_quad:
                class_dict_quad[tuple(np.array(label))] = []
            if pixel_to_sum_quad[pixel] !=0:
                class_dict_quad[tuple(np.array(label))].append(pixel_to_sum_quad[pixel])

        if len(pixel_to_label_new_quad)>0:
            for pixel, label in pixel_to_label_new_quad.items():
                if tuple(np.array(label)) not in class_dict_quad:
                    class_dict_quad[tuple(np.array(label))] = []
                if pixel_to_sum_quad[pixel] != 0:
                    class_dict_quad[tuple(np.array(label))].append(pixel_to_sum_quad[pixel])




    print('Generated class distribution: \n')
    for aclass in class_dict:

        print(aclass, 'non_Zero: ', non_zero_dict(class_dict[aclass]), '\n')
        print(aclass, ': ', len(class_dict[aclass]), '\n')
    if not Quarter_Fill:
        print('Generated class distribution quad: \n')
        for aclass in class_dict_quad:
            print(aclass, 'non_Zero: ', non_zero_dict(class_dict_quad[aclass]), '\n')

            print(aclass, ': ', len(class_dict_quad[aclass]), '\n')

    # Save uniquely generated samples per class based on a dictionary
    for aclass in class_dict:
        print(class_dict[aclass])
        unique_generated = sum([1 for val in class_dict[aclass] if val > 0])
        ex.log_scalar(str(aclass), unique_generated, epoch)
    if not Quarter_Fill:
    # Save uniquely generated samples per class based on a dictionary
        for aclass in class_dict_quad:
            print(class_dict_quad[aclass])

            unique_generated = sum([1 for val in class_dict_quad[aclass] if val > 0])
            ex.log_scalar(str(aclass) + 'quad', unique_generated, epoch)

    # key = list(class_dict.keys())[0]
    # num_label = len(class_dict[key])
    # plt.bar([i for i in range(0, num_label)], class_dict[key])
    # plt.show()

    start = 0
    for idx, aclass in enumerate(class_dict):
        num_label = len(class_dict[aclass])
        sum_label = sum(class_dict[aclass])
        start += num_label
        # What percentage of possible configurations are generated
        unique_generated = sum([1 for val in class_dict[aclass] if val > 0])
        print('Configurations generated for class: ', aclass, ' is :',
              unique_generated, 'out of: ', num_label, ' ', sum_label,'\n')

        ex.log_scalar(','.join([str(i) for i in aclass]), unique_generated, epoch)

    id = ex.current_run._id


    # Histogram of a histogram
    freq_every_pixel = [freq for aclass in class_dict for freq in class_dict[aclass]]
    count_freq = defaultdict(lambda: 0)
    for freq in freq_every_pixel:
        count_freq[freq] += 1


    print('{} plots and samples generated in {} seconds.'.format(num_test_samples, time.time() - start_time))
    if not Quarter_Fill:
        start = 0
        for idx, aclass in enumerate(class_dict_quad):
            num_label = len(class_dict_quad[aclass])
            sum_label = sum(class_dict_quad[aclass])

            start += num_label
            # What percentage of possible configurations are generated
            unique_generated = sum([1 for val in class_dict_quad[aclass] if val > 0])
            print('Configurations generated for class quad: ', aclass, ' is :',
                  unique_generated, 'out of: ', num_label, ' ', sum_label, '\n')

            ex.log_scalar(','.join([str(i) for i in aclass]), unique_generated, epoch)

        id = ex.current_run._id
        # plt.savefig('logs/{}/generated_class_distribution_{}_samples_quad.png'.format(id, num_test_samples))
        # plt.clf()
        # with open('logs/{}/class_distribution_quad.pickle'.format(id), 'wb') as handle:
        #     pickle.dump(class_dict_quad, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # Histogram of a histogram
        freq_every_pixel = [freq for aclass in class_dict_quad for freq in class_dict_quad[aclass]]
        count_freq = defaultdict(lambda: 0)
        for freq in freq_every_pixel:
            count_freq[freq] += 1

        # plt.bar(list(count_freq.keys()), list(count_freq.values()))
        # plt.ylim(0, 50)
        # plt.xlim(0, 1000)
        # plt.title('Histogram of the generated class distribution')
        # plt.savefig('logs/{}/hist_of_generated_class_distribution_{}_samples_quad.png'.format(id, num_test_samples))
        # plt.clf()
        print('{} plots and samples generated in {} seconds.'.format(num_test_samples, time.time() - start_time))
    return


def evaluate_generation_distribution(ex, epoch,test_images, categorization, cat_names, generator, pixel_to_label, num_condition,
                                 full_image, pixel_len, Quarter_Fill, zdim,
                                     accuracy_matrix, class_accuracies, accuracy_matrix_quad, class_accuracies_quad,
                                     accuracy_matrix_without_unknowns,accuracy_matrix_without_unknowns_quad,
                                     conditioned_labels, lookup_labels, lookup_labels_quad, num_unknowns,
                                 plot_generated_distribution=False):

    print('accuracy_matrix ', accuracy_matrix)
    print('accuracy_matrix ', accuracy_matrix_quad)

    print('accuracy_matrix without unknowns ', accuracy_matrix_without_unknowns)
    print('accuracy_matrix without unknowns ', accuracy_matrix_without_unknowns_quad)
    quad = True
    uniqueValues, occurCount = np.unique(test_images, axis=0, return_counts=True)
    ex.log_scalar('Total uniquely generated samples', len(uniqueValues))
    pixel_to_label_new = {}
    pixel_to_label_new_quad = {}
    if True:
        pixel_to_sum = {pixel: 0 for pixel in pixel_to_label}  # Count number of times each pixel appeared
        pixel_to_sum_quad = {pixel: 0 for pixel in pixel_to_label}  # Count number of times each pixel appeared

        pixel_to_sum_convert = {convert_tuple_binary_to_int(pixel): 0 for pixel in
                                pixel_to_label}  # Count number of times each pixel appeared
        pixel_to_sum_quad_convert = {convert_tuple_binary_to_int(pixel): 0 for pixel in pixel_to_label}
    for index, test_image in enumerate(test_images):
        if Quarter_Fill:
            test_image = quad_from_partial(test_image.reshape(pixel_len, pixel_len))
        else:
            test_image = test_image.reshape(pixel_len * pixel_len)
            if quad:
                test_image1 = quad_from_partial(test_image.reshape(pixel_len, pixel_len))
                test_image1 = get_partial_three_fold_from_quad(test_image1, pixel_len)
                test_image1 = test_image1.reshape(pixel_len * pixel_len)
                if tuple(test_image1) in pixel_to_sum_quad:
                    pixel_to_sum_quad[tuple(test_image1)] += 1
                    pixel_to_sum_quad_convert[convert_tuple_binary_to_int(test_image1)] += 1
                else:
                    if lookup_labels_quad[index] != 'unknown':
                        if lookup_labels_quad[index] == 0:
                            pixel_to_label_new_quad[tuple(test_image1)] = torch.tensor([1, 0])
                        else:
                            pixel_to_label_new_quad[tuple(test_image1)] = torch.tensor([0, 1])
                        pixel_to_sum_quad[tuple(test_image1)] = 1  # New pixel
                        pixel_to_sum_quad_convert[convert_tuple_binary_to_int(test_image1)] = 1

        if tuple(test_image) in pixel_to_sum:
            pixel_to_sum[tuple(test_image)] += 1
            pixel_to_sum_convert[convert_tuple_binary_to_int(test_image)] += 1

        else:
            if lookup_labels[index] != 'unknown':
                pixel_to_sum[tuple(test_image)] = 1
                pixel_to_sum_convert[convert_tuple_binary_to_int(test_image)] = 1

                if lookup_labels[index] == 0:
                    pixel_to_label_new[tuple(test_image)] = torch.tensor([1, 0])
                else:
                    pixel_to_label_new[tuple(test_image)] = torch.tensor([0, 1])

    class_dict = {}
    class_dict_quad = {}
    class_dict_pixels = {}
    class_dict_pixels_quad = {}

    for pixel, label in pixel_to_label.items():
        if tuple(np.array(label)) not in class_dict:
            class_dict[tuple(np.array(label))] = []
        if pixel_to_sum[pixel] != 0:
            class_dict[tuple(np.array(label))].append(pixel_to_sum[pixel])
    if len(pixel_to_label_new) > 0:
        for pixel, label in pixel_to_label_new.items():
            if tuple(np.array(label)) not in class_dict:
                class_dict[tuple(np.array(label))] = []
            if pixel_to_sum[pixel] != 0:
                class_dict[tuple(np.array(label))].append(pixel_to_sum[pixel])
    if not Quarter_Fill:
        for pixel, label in pixel_to_label.items():
            if tuple(np.array(label)) not in class_dict_quad:
                class_dict_quad[tuple(np.array(label))] = []
            if pixel_to_sum_quad[pixel] != 0:
                class_dict_quad[tuple(np.array(label))].append(pixel_to_sum_quad[pixel])

        if len(pixel_to_label_new_quad) > 0:
            for pixel, label in pixel_to_label_new_quad.items():
                if tuple(np.array(label)) not in class_dict_quad:
                    class_dict_quad[tuple(np.array(label))] = []
                if pixel_to_sum_quad[pixel] != 0:
                    class_dict_quad[tuple(np.array(label))].append(pixel_to_sum_quad[pixel])

    print('Generated class distribution: \n')
    for aclass in class_dict:
        print(aclass, 'non_Zero: ', non_zero_dict(class_dict[aclass]), '\n')
        print(aclass, ': ', len(class_dict[aclass]), '\n')
    if not Quarter_Fill:
        print('Generated class distribution quad: \n')
        for aclass in class_dict_quad:
            print(aclass, 'non_Zero: ', non_zero_dict(class_dict_quad[aclass]), '\n')

            print(aclass, ': ', len(class_dict_quad[aclass]), '\n')

    # Save uniquely generated samples per class based on a dictionary
    for aclass in class_dict:
        print(class_dict[aclass])
        unique_generated = sum([1 for val in class_dict[aclass] if val > 0])
        ex.log_scalar(str(aclass), unique_generated, epoch)
    if not Quarter_Fill:
        # Save uniquely generated samples per class based on a dictionary
        for aclass in class_dict_quad:
            # print(class_dict_quad[aclass])

            unique_generated = sum([1 for val in class_dict_quad[aclass] if val > 0])
            ex.log_scalar(str(aclass) + 'quad', unique_generated, epoch)

    start = 0

    for idx, aclass in enumerate(class_dict):
        num_label = len(class_dict[aclass])
        sum_label = sum(class_dict[aclass])
        start += num_label
        # What percentage of possible configurations are generated
        unique_generated = sum([1 for val in class_dict[aclass] if val > 0])
        print('Configurations generated for class: ', aclass, ' is :',
              unique_generated, 'out of: ', num_label, ' ', sum_label, '\n')

        ex.log_scalar(','.join([str(i) for i in aclass]), unique_generated, epoch)


    if not Quarter_Fill:
        start = 0

        for idx, aclass in enumerate(class_dict_quad):
            num_label = len(class_dict_quad[aclass])
            sum_label = sum(class_dict_quad[aclass])

            start += num_label
            # What percentage of possible configurations are generated
            unique_generated = sum([1 for val in class_dict_quad[aclass] if val > 0])
            print('Configurations generated for class quad: ', aclass, ' is :',
                  unique_generated, 'out of: ', num_label, ' ', sum_label, '\n')

            ex.log_scalar(','.join([str(i) for i in aclass]), unique_generated, epoch)

    return
