
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
import pickle
import time
from matplotlib import pyplot
from helper import *



class Logger:

    def __init__(self, model_name, data_name, id, image_size, dir_summary=None):

      self.model_name = model_name
      self.data_name = data_name
      self.id = id
      self.image_size = image_size

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

    def log(self, d_error, g_error, epoch, n_batch, num_batches, r_error=None):

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

    def save_images(self, test_images, epoch, i, dimension, level=2, conditions=[1, 0],iso_condition=0):
      temp = edit_output_image(test_images, level=level, conditions=conditions)
      fake_points = get_points_from_matrix(temp, condition=conditions[1-iso_condition])
      fake_points1 = []
      if level == 3:
        fake_points1 = get_points_from_matrix(temp, condition=conditions[iso_condition])

      out_dir = './results/{}/'.format(self.id)
      Logger._make_dir(out_dir)
      if len(fake_points) > 0:
        visualize_tumor_3d(fake_points[:, 0], fake_points[:, 1], fake_points[:, 2], epoch=epoch, id=i,
                           set_lim=True, lim=dimension, color='red', save=True, dir=out_dir)
      if len(fake_points1) > 0:
        visualize_tumor_3d(fake_points1[:, 0], fake_points1[:, 1], fake_points1[:, 2], epoch=epoch,
                           id=(str(i) + 'iso'),
                           set_lim=True, lim=dimension, color='blue', save=True, dir=out_dir)
      # fig.savefig('{}/results_epoch_{}.png'.format(out_dir, epoch))
      # self.log_figure('results', fig, epoch)

    def save_images_iso(self, test_images, epoch, i, dimension, level=2, conditions = [1,0],iso_condition=0,out_dir=''):
      temp = edit_output_image(test_images, level=level, conditions = conditions)
      fake_points = get_points_from_matrix(temp, condition=conditions[1-iso_condition])
      fake_points1 = []
      if level  == 3:
        fake_points1 = get_points_from_matrix(temp, condition=conditions[iso_condition])

      if out_dir =='':
        out_dir = './results/{}/'.format(self.id)
        Logger._make_dir(out_dir)
      if len(fake_points)>0:
        visualize_tumor_3d(fake_points[:, 0], fake_points[:, 1], fake_points[:, 2], epoch=epoch, id=i,
                           set_lim=True, lim=dimension, color='red', save=True, dir=out_dir)
      if len(fake_points1) > 0:
        visualize_tumor_3d(fake_points1[:, 0], fake_points1[:, 1], fake_points1[:, 2], epoch=epoch, id=(str(i)+'_iso'),
                           set_lim=True, lim=dimension, color='blue', save=True, dir=out_dir)
      # fig.savefig('{}/results_epoch_{}.png'.format(out_dir, epoch))
      # self.log_figure('results', fig, epoch)

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

    def display_status(self, epoch, num_epochs, n_batch, num_batches, d_error, g_error, d_pred_real, d_pred_fake):

      # var_class = torch.autograd.variable.Variable
      if isinstance(d_error, torch.autograd.Variable):
        d_error = d_error.data.cpu().numpy()
      if isinstance(g_error, torch.autograd.Variable):
        g_error = g_error.data.cpu().numpy()
      if isinstance(d_pred_real, torch.autograd.Variable):
        d_pred_real = d_pred_real.data
      if isinstance(d_pred_fake, torch.autograd.Variable):
        d_pred_fake = d_pred_fake.data

      print('Epoch: [{}/{}], Batch Num: [{}/{}]'.format(
        epoch, num_epochs, n_batch, num_batches)
      )
      print('Discriminator Loss: {:.4f}, Generator Loss: {:.4f}'.format(d_error, g_error))
      print('D(x): {:.4f}, D(G(z)): {:.4f}'.format(d_pred_real.mean(), d_pred_fake.mean()))

    def save_models(self, generator, discriminator, epoch):
      out_dir = './data/models/{}'.format(self.data_subdir)
      Logger._make_dir(out_dir)
      torch.save(generator.state_dict(),
                 '{}/G_epoch_{}'.format(out_dir, epoch))
      torch.save(discriminator.state_dict(),
                 '{}/D_epoch_{}'.format(out_dir, epoch))

    
    def log_metrics(self, ex, result_dict, epoch, dir=None):
      if dir==None:
        out_dir = './logs/{}/'.format(self.id)
        Logger._make_dir(out_dir)
      else:
        out_dir = dir
      with open(out_dir + 'result_dict_epoch'+ str(epoch)+ '.pkl', 'wb') as handle:
          pickle.dump(result_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

      # print('Logging metrics')
      # for itr, res in result_dict.items():
      #   for name_object, counts in res.items():
      #       for index, t in enumerate(counts):
      #         self.writer.add_scalar('{}/{}'.format(self.comment, str(itr) +'_'+ name_object+'_' + str(index)), t, epoch)
      #         ex.log_scalar( str(itr)+'_' + name_object+'_' + str(index), t, epoch)

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


