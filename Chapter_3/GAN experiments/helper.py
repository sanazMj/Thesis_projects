
import numpy as np
import matplotlib.ticker as ticker
import pandas as pd
import h5py
import json
import os
import pickle
import matplotlib.pyplot as plt
import plotly.offline as off_line
from IPython.core.pylabtools import figsize
import plotly.figure_factory as ff
import plotly.graph_objects as go
import random
import numpy as np
from scipy.spatial import Delaunay
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import torch.autograd as autograd
from torch.autograd.variable import Variable
from tabulate import tabulate
import math
import scipy.io as sio
from scipy.ndimage.measurements import label
from scipy.ndimage import generate_binary_structure
from scipy.spatial import ConvexHull, convex_hull_plot_2d,  distance
from sklearn.cluster import KMeans
from sklearn import metrics
from skimage.measure import label as Label_measure
from skimage.measure import  regionprops, regionprops_table
from discrete_frechet_master.distances.discrete import DiscreteFrechet, LinearDiscreteFrechet, VectorizedDiscreteFrechet
from discrete_frechet_master.distances.discrete import euclidean

from constants import Constant_dict, Constant_dict_keys


def plot_trisurf_matlab(points):
    '''
    Function is used to plot the mesh figure of a set of points
    :param points: set of points
    :return: Mesh plot of points
    '''
    fig = plt.figure(figsize=(8, 5))
    ax = plt.axes(projection='3d')

    hull = ConvexHull(points)

    # ax.plot(X, Y, Z, 'bo', ms=2)
    ax.plot(points[hull.vertices, 0],
            points[hull.vertices, 1],
            points[hull.vertices, 2], 'ko', markersize=4)
    s = ax.plot_trisurf(points[:,0], points[:,1], points[:,2], triangles=hull.simplices,
                        cmap='viridis', alpha=0.2, edgecolor='k')
    # plt.colorbar(s, shrink=0.7)

    plt.show()
    return

def moment(points, a1, a2, a3):
    '''
    moment of a set of points
    :param points: set of points
    :param a1: power hyperparameter for the first coordinate
    :param a2:  power hyperparameter for the second coordinate
    :param a3:  power hyperparameter for the third coordinate
    :return: moment
    '''
    result = 0
    for x,y,z in points:
        result += x**a1 * y**a2 * z **a3
    return result

def moment_invariants(points):
    '''

    :param points: set of input points
    :return: 3D moment invariants of the points
    '''
    vol = moment(points, 0, 0, 0)
    temp200 =  moment(points, 2, 0, 0)
    temp020 =  moment(points, 0, 2, 0)
    temp002 =  moment(points, 0, 0, 2)
    temp110 =  moment(points, 1, 1, 0)
    temp101 =  moment(points, 1, 0, 1)
    temp011 =  moment(points, 0, 1, 1)
    non_normalized_moment1 = temp200 + temp020 + temp002
    non_normalized_moment2 = temp200 * temp020 +  temp200*temp002 +  temp020*temp002 -temp110**2 -temp101**2 -temp011**2
    non_normalized_moment3 =  temp200 * temp020* temp002 + 2* temp110 * temp101 * temp011 - temp200*temp011**2 -temp020*temp101**2 -temp002 * temp110**2
    result = [(3* vol ** (5/3))/non_normalized_moment1,
              (3* vol ** (10/3))/non_normalized_moment2,
              (vol ** 5)/non_normalized_moment3]
    return result

def connected_target_1(y, landa1 = 1, landa2 =10):
    '''
    connectivity loss for a set of samples y with no target isocenters
    :param y: samples
    :param landa1: hyperparameter
    :param landa2: hyperparameter
    :return:  connectivity loss
    '''
    s = generate_binary_structure(3, 2)
    Loss_all = []
    y = y.detach().cpu().numpy()
    y = y.reshape(y.shape[0]*y.shape[1], y.shape[2], y.shape[3], y.shape[4])
    for i in y:
        x = np.where(i>=0.5,1,0)
        labeled_array_tumor, tumor_count = label(x, structure=s)
        Loss = np.abs(tumor_count - 1) - landa1 * int(tumor_count == 1)
        Loss = Loss/landa2
        Loss_all.append(Loss)
    b = torch.tensor(np.array(Loss_all))
    b = autograd.Variable(b.cuda(), requires_grad=True)
    b = (torch.mean(b))
    return b

def connected(y, target_points=1,dataset_include_tumor=False, landa1 = 1):
    '''
    connectivity loss for a set of samples y with # target_points isocenters
    :param y: samples
    :param target_points: number of target isocenters
    :param dataset_include_tumor: whether the dataset include tumors as well
    :param landa1: hyperparameter
    :return: connectivity loss
    '''
    s = generate_binary_structure(3, 2)
    targetpoints = target_points + int(dataset_include_tumor)
    Loss_all = []
    y = y.detach().cpu().numpy()
    y = y.reshape(y.shape[0]*y.shape[1], y.shape[2], y.shape[3], y.shape[4])
    for i in y:
        tumor_count_list = []
        x = np.floor(i * targetpoints)
        range_list = [i for i in np.unique(x) if i != int(dataset_include_tumor) and i != 0]
        if len(range_list) > 0:
            for k in range_list:
                temp = np.where(x == k, 1, 0)
                labeled_array_tumor, tumor_count = label(temp, structure=s)
                tumor_count_list.append(tumor_count)
            tumor_count_list = np.array(tumor_count_list)
            Loss = np.sqrt(np.mean((tumor_count_list - 1) ** 2)) - landa1 * np.count_nonzero(tumor_count_list == 1)
        else:
            Loss = 0
        Loss = Loss/10
        Loss_all.append(Loss)
    b = torch.tensor(np.array(Loss_all))
    b = autograd.Variable(b.cuda(), requires_grad=True)
    b = (torch.mean(b))
    return b

def sofSparsity(x, alpha=1, beta=1):
    '''
    returns the softsparsity of a sample
    :param x: input sample
    :param alpha: hyperparamter
    :param beta: hyperparamter
    :return:
    '''
    result = []

    x = x.reshape(x.shape[0], -1)
    for data in x:
        temp = 0
        for j in data:
            if j !=0:
                nom = j**alpha + (1/j)**beta
            temp += np.abs(nom/(nom+1))
        result.append(temp/x.shape[1])
    return result


def shannon_formula( val, count, include_tumor=False, target_points=None):
    '''
    returns the Shannon index
    :param val: list of unique values
    :param count: frequency of values
    :param include_tumor: whether samples include tumor
    :param target_points: number of target isocenters
    :return:  Shannon index
    '''
    if include_tumor:
        val_list =[int(i) for i in val if int(i) != 0 and int(i) != 1]
        count_list = [count[i] for i in range(len(val)) if int(val[i]) != 0 and int(val[i]) != 1]
    else:
        val_list = [int(i) for i in val if int(i) != 0]
        count_list = [count[i] for i in range(len(val)) if int(val[i]) != 0]
    if target_points==None:
        target_points = np.max(val_list)

    count_list_new = [1 for i in range(target_points)]

    for index, j in enumerate(val_list):

        count_list_new[int(j - 1- int(include_tumor))] += count_list[index]
    temp = count_list_new / np.sum(count_list_new)
    p_sum = 0
    for index, p in enumerate(temp):
        p_sum += -1 * p * math.log(p, 2)
    result = (p_sum / math.log(len(temp), 2))
    return result

def shannon_diversity(data, target_points):
    result = []
    step = 1 / (target_points+1)
    dimension = [data.shape[2],data.shape[3], data.shape[4]]
    data = data.reshape(data.shape[0]*data.shape[1], data.shape[2],data.shape[3], data.shape[4] )

    for x in data:
        # temp1 = np.zeros(shape=(dimension[0], dimension[1], dimension[2]))
        # for j in range(1, target_points + 1):
        #     temp1[np.where(((j * step) <= x) & (x < (step * (j + 1))))] = j
        temp1 = np.floor(x * (target_points + 1))
        temp1 = np.where(temp1 == (target_points + 1), target_points, temp1)
        count_list = [1 for i in range(target_points)]
        val, count = np.unique(temp1,return_counts=True)
        for index, j in enumerate(val):
            count_list[int(j-1)] += count[index]
        temp = count_list/np.sum(count_list)
        p_sum = 0
        for index, p in enumerate(temp):
            p_sum += -1 * p * math.log(p,2)
        result.append(p_sum/math.log(len(temp),2))
    return result

def set_values_in_array(temp, indexes, value):
    index_list = []
    for i in range(indexes.shape[1]):
        index_list.append(indexes[:,i])
    temp[index_list] = value
    return temp

def set_values_in_dict(arguments, names):
    result = {}
    for index, arg in enumerate(arguments):
        result[names[index]] = arg
    return result

def list_unique(temp):
    return_list =[]
    for i in range(len(temp)):
        return_list.append(list(temp[i]))
    value_unique, count_unique = np.unique(return_list, return_counts=True)
    return value_unique, count_unique

def print_results(results_dict,keys=[],show=['mean','std']):
    return_dict = {}
    key_list = keys if len(keys)>0 else list(results_dict.keys())
    for key in key_list:
        if key in list(results_dict.keys()):
            print(key)

            return_dict[key + '_mean'] = np.mean(np.array(results_dict[key]))
            return_dict[key + '_std'] = np.std(np.array(results_dict[key]))

            if 'var' in show:
                print(key, results_dict[key], np.mean(np.array(results_dict[key])), np.std(np.array(results_dict[key])))
            else:
                print(key, np.mean(np.array(results_dict[key])), np.std(np.array(results_dict[key])))
    return return_dict

def coef_func(x):
    if x >100:
        return x * 0.00001
    else:
        return x * 0.01

def Plotly_trisurf(temp, color, points=None):
    if not points:
        points = get_points_from_matrix(temp, condition=1)
    if points.shape[0] > 4:
        x = points[:,0]
        y = points[:,1]
        z = points[:,2]
        mesh = go.Mesh3d(x=x,y=y,z=z,alphahull=0,opacity=0.2,color=color)
        return mesh
    else:
        return []
def Matlab_trisurf(temp, file_name, conditions, points=None):
    for condition in conditions:
        if not points:
            points = get_points_from_matrix(temp, condition=condition)
        if points.shape[0] > 4:
            hull = ConvexHull(points)
            # print(hull.simplices, hull.vertices)
            x = points[:,0]
            y = points[:,1]
            z = points[:,2]
            fig = ff.create_trisurf(x=x, y=y, z=z,
                                     colormap=['rgb(50, 0, 75)', 'rgb(200, 0, 200)', 'rgb(10, 0, 200)','rgb(50, 0, 200)','rgb(120, 0, 200)','rgb(140, 0, 200)','#c8dcc8'],
                                     show_colorbar=True,
                                     simplices=hull.simplices,
                                     title="Boy's Surface")
        # fig.show()
    off_line.iplot(fig,show_link=False)
    off_line.plot(fig, auto_open=False, filename=file_name + '.html')

def edit_output_image(generated_sample, level=2, conditions = [1,2], activation='sigmoid'):
    '''
    Post-process the generated image
    :param generated_sample: sample input
    :param level: how many level conditions are there
    :param conditions: the thresholds
    :param activation: what is the activation of the generated samples
    :return:
    '''
    generated_sample = generated_sample.cpu()
    if len(generated_sample.shape)>4:
        temp = conditions[0] * np.ones((generated_sample.shape[0], 1, generated_sample.shape[2], generated_sample.shape[3], generated_sample.shape[4]))
    else:
        temp = conditions[0] * np.ones((generated_sample.shape[0],generated_sample.shape[1], generated_sample.shape[2], generated_sample.shape[3]))
    if level==2 and activation=='sigmoid':
        temp[np.where(generated_sample < 0.5)] = 0
    elif level==2 and activation=='Tanh':
        temp[np.where(generated_sample < 0)] = 0
    elif level == 3:
        temp[np.where(generated_sample < 0.33)] = 0
        temp[np.where(generated_sample > 0.66)] = conditions[1]
    return temp

def get_points_from_matrix(temp,condition=1, dim_count=3):
    '''
    Returns the coordinates of points inside a matrix equal to condition
    :param temp: input
    :param condition:
    :param dim_count: dimension
    :return:
    '''
    temp_indice_specific = np.where(temp==condition)
    points = []
    if len(temp_indice_specific[0])>0:
        tempsize = temp_indice_specific[0].shape[0]
        if len(temp.shape)==3:
            a = temp_indice_specific[0].reshape(tempsize, 1)
            b = temp_indice_specific[1].reshape(tempsize, 1)
            c =  temp_indice_specific[2].reshape(tempsize, 1)
            points = np.concatenate((a, b, c), axis=1)

        elif dim_count==3:
            a = temp_indice_specific[1].reshape(tempsize, 1)
            b = temp_indice_specific[2].reshape(tempsize, 1)
            c = temp_indice_specific[3].reshape(tempsize, 1)
            points = np.concatenate((a, b, c), axis=1)

        elif dim_count==4:
            a1 = temp_indice_specific[0].reshape(tempsize, 1)
            a = temp_indice_specific[1].reshape(tempsize, 1)
            b = temp_indice_specific[2].reshape(tempsize, 1)
            c = temp_indice_specific[3].reshape(tempsize, 1)
            points = np.concatenate((a1, a, b, c), axis=1)

    return points



def visualize_multi_colored_isos_edited(temp,name, epoch, id,views, lim, set_lim=False, save=False,
                     dir='/home/sanaz/Ryerson/Projects/TumorGAN/GAN_3D/3D_chair_results/'):
    '''
    Save or plot the scatter 3D of an image from different views
    :param temp: input matrix
    :param name: output name
    :param epoch:
    :param id:
    :param views: list of views to illustrate the fig
    :param lim: limits of image
    :param set_lim: True if you want to set a limit
    :param save: True if you want to save the image
    :param dir: save directory
    :return:
    '''
    # rs = np.random.rand(len(np.unique(temp)), 3)
    colors_list = ['r','g','b','c', 'm', 'y', 'k', 'navy', 'coral', 'cyan', 'springgreen', 'lightgray']
    count = len(np.unique(temp))

    ax1 = plt.axes(projection='3d')

    if len(views) == 0:
        views = [(-1,-1)]
    for view in views:
        # ax = plt.axes(projection='3d')
        for index, i in enumerate(list(np.unique(temp))):
            if i !=0:

                points = get_points_from_matrix(temp, int(i))
                print(i, len(points), index)

                if len(points) > 0:
                    ax1.scatter3D(points[:, 0], points[:, 1], points[:, 2], color=colors_list[index])

                # plt.show()
        if set_lim:
            ax1.set_xlim(0, lim[0])
            ax1.set_ylim(0, lim[1])
            ax1.set_zlim(0, lim[2])
        if save:

            plt.savefig(dir + name + str(epoch) + '_' + str(id) + '.png')
        else:
            if view[0] != -1:
                ax.view_init(view[0], view[1])
            plt.show()

def visualize_multi_colored_isos_one_image(temp,name, epoch, id,views, lim, x_lim_fig=5, y_lim_fig=10,set_lim=False, voxel=False, trisurf=False, save=False, add_tumor=False,
                     dir='/home/sanaz/Ryerson/Projects/TumorGAN/GAN_3D/3D_chair_results/'):
    '''
      Save or plot the scatter/voxel/trisurf 3D of an image from different views
      :param temp: input matrix
      :param name: output name
      :param epoch:
      :param id:
      :param views: list of views to illustrate the fig
      :param lim: limits of image
      :param voxel:
      :param trisurf
      :param set_lim: True if you want to set a limit
      :param save: True if you want to save the image
      :param dir: save directory
      :return:
      '''
    colors_list = ['r','g','b','c', 'm', 'y', 'k', 'navy', 'coral', 'cyan', 'springgreen', 'lightgray']
    count = len(np.unique(temp))
    # if add_tumor:
    #     count -=1
    fig = plt.figure(figsize=(1*x_lim_fig, y_lim_fig))
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    temp2 = np.where(temp != 0, 1, 0)
    points_temp = get_points_from_matrix(temp2, 1)
    if len(views) == 0:
        views = [(-1,-1)]
    for view in views:
        if trisurf:

                hull = ConvexHull(points_temp)
                surf = ax.plot_trisurf(points_temp[:, 0], points_temp[:, 1], points_temp[:, 2], triangles=hull.simplices,
                                     color='springgreen', alpha=0.1, edgecolor='k', label='Tumor')
                ax.xaxis.set_major_locator(ticker.NullLocator())
                ax.yaxis.set_major_locator(ticker.NullLocator())
                ax.zaxis.set_major_locator(ticker.NullLocator())
                surf._edgecolors2d = surf._edgecolors3d
                surf._facecolors2d = surf._facecolors3d
        else:
                surf = ax.voxels(temp2, edgecolor='k', facecolors='springgreen', alpha=0.2, label='Tumor')
                ax.xaxis.set_major_locator(ticker.NullLocator())
                ax.yaxis.set_major_locator(ticker.NullLocator())
                ax.zaxis.set_major_locator(ticker.NullLocator())
                surf._edgecolors2d = surf._edgecolors3d
                surf._facecolors2d = surf._facecolors3d

        for index, i in enumerate(list(np.unique(temp))):
            if i !=0 and i!=1:

                points = get_points_from_matrix(temp, int(i))
                temp1 = np.where(temp== int(i), 1, 0)
                if len(points) > 0:
                    if voxel:
                        surf= ax.voxels(temp1, edgecolor='k', facecolors=colors_list[index], alpha=0.2, label='Iso '+ str(i-1))
                        ax.xaxis.set_major_locator(ticker.NullLocator())
                        ax.yaxis.set_major_locator(ticker.NullLocator())
                        ax.zaxis.set_major_locator(ticker.NullLocator())
                        surf._edgecolors2d = surf._edgecolors3d
                        surf._facecolors2d = surf._facecolors3d

                    elif trisurf:
                        hull = ConvexHull(points)

                        surf= ax.plot_trisurf(points[:, 0], points[:, 1], points[:, 2], triangles=hull.simplices,
                                            color=colors_list[index], alpha=0.2, edgecolor='k', label='Iso' + str(i-1))
                        surf._edgecolors2d = surf._edgecolors3d
                        surf._facecolors2d = surf._facecolors3d

                        ax.xaxis.set_major_locator(ticker.NullLocator())
                        ax.yaxis.set_major_locator(ticker.NullLocator())
                        ax.zaxis.set_major_locator(ticker.NullLocator())
                    else:
                        ax.scatter3D(points[:, 0], points[:, 1], points[:, 2], color=colors_list[index])
                    # if add_tumor and int(i) == 1:
                    #     # ax.set_xlabel('tumor', fontsize=10)
                    #     ax.set_title('Tumor', fontsize=15)
                    # else:
                    #     ax.set_title('Iso '+str(int(i-1)), fontsize=15)

                        # ax.set_xlabel('iso '+str(int(i-1)), fontsize=10)
                    if set_lim:
                        ax.set_xlim(0, lim[0])
                        ax.set_ylim(0, lim[1])
                        ax.set_zlim(0, lim[2])

        # ax.legend()
        if save:
            ax.legend(loc='upper right', bbox_to_anchor=(1.25,1))
            plt.savefig( dir + name + str(epoch) + '_' + str(id) + '.pdf', bbox_inches='tight')
        else:
            if view[0] != -1:
                ax.view_init(view[0], view[1])
            plt.show()

def visualize_multi_colored_isos(temp,name, epoch, id,views, lim, x_lim_fig=5, y_lim_fig=10,set_lim=False, voxel=False, trisurf=False, save=False, add_tumor=False,
                     dir='/home/sanaz/Ryerson/Projects/TumorGAN/GAN_3D/3D_chair_results/'):
    # rs = np.random.rand(len(np.unique(temp)), 3)
    colors_list = ['r','g','b','c', 'm', 'y', 'k', 'navy', 'coral', 'cyan', 'springgreen', 'lightgray']
    count = len(np.unique(temp))
    # if add_tumor:
    #     count -=1
    fig = plt.figure(figsize=(count*x_lim_fig, y_lim_fig))
    ax1 = fig.add_subplot(1, count, 1, projection='3d')
    if len(views) == 0:
        views = [(-1,-1)]
    for view in views:
        if trisurf:
            temp2 = np.zeros((temp.shape[0], temp.shape[1], temp.shape[2]))
            temp2 = np.where(temp != 0, 1, 0)
            points_temp = get_points_from_matrix(temp2, 1)
            hull = ConvexHull(points_temp)

            s = ax1.plot_trisurf(points_temp[:, 0], points_temp[:, 1], points_temp[:, 2], triangles=hull.simplices,
                                cmap='viridis', alpha=0.2, edgecolor='k')
            ax1.xaxis.set_major_locator(ticker.NullLocator())
            ax1.yaxis.set_major_locator(ticker.NullLocator())
            ax1.zaxis.set_major_locator(ticker.NullLocator())

        # ax = plt.axes(projection='3d')
        temp1 = np.zeros((temp.shape[0], temp.shape[1], temp.shape[2]))
        for index, i in enumerate(list(np.unique(temp))):
            if i !=0 :
                # ax = plt.axes(projection='3d')
                ax = fig.add_subplot(1, count, index+1, projection='3d')
                points = get_points_from_matrix(temp, int(i))
                temp1 = np.where(temp== int(i), 1, 0)
                # colorg = (rs[index,0],rs[index,1],rs[index,2])
                if len(points) > 0:
                    if voxel:
                        ax.voxels(temp1, edgecolor='k', facecolors=colors_list[index])
                        ax.xaxis.set_major_locator(ticker.NullLocator())
                        ax.yaxis.set_major_locator(ticker.NullLocator())
                        ax.zaxis.set_major_locator(ticker.NullLocator())
                    elif trisurf:
                        hull = ConvexHull(points)

                        # ax.plot(X, Y, Z, 'bo', ms=2)
                        # ax.plot(points[hull.vertices, 0],
                        #         points[hull.vertices, 1],
                        #         points[hull.vertices, 2], 'ko', markersize=4)
                        s = ax.plot_trisurf(points[:, 0], points[:, 1], points[:, 2], triangles=hull.simplices,
                                            cmap='viridis', alpha=0.2, edgecolor='k')
                        ax.xaxis.set_major_locator(ticker.NullLocator())
                        ax.yaxis.set_major_locator(ticker.NullLocator())
                        ax.zaxis.set_major_locator(ticker.NullLocator())
                    else:
                        ax.scatter3D(points[:, 0], points[:, 1], points[:, 2], color=colors_list[index])

                    if set_lim:
                        ax.set_xlim(0, lim[0])
                        ax.set_ylim(0, lim[1])
                        ax.set_zlim(0, lim[2])
                    if voxel:
                        ax1.voxels(temp1, edgecolor='k', facecolors=colors_list[index], label=str(int(i-1)))
                        ax1.xaxis.set_major_locator(ticker.NullLocator())
                        ax1.yaxis.set_major_locator(ticker.NullLocator())
                        ax1.zaxis.set_major_locator(ticker.NullLocator())

                    elif trisurf==False:
                        ax1.scatter3D(points[:, 0], points[:, 1], points[:, 2], color=colors_list[index])
                        ax1.xaxis.set_major_locator(ticker.NullLocator())
                        ax1.yaxis.set_major_locator(ticker.NullLocator())
                        ax1.zaxis.set_major_locator(ticker.NullLocator())
                # plt.show()
        if set_lim:
            ax1.set_xlim(0, lim[0])
            ax1.set_ylim(0, lim[1])
            ax1.set_zlim(0, lim[2])
        ax1.legend()
        if save:

            plt.savefig(dir + name + str(epoch) + '_' + str(id) + '.pdf', bbox_inches='tight')
        else:
            if view[0] != -1:
                ax.view_init(view[0], view[1])
            plt.show()

def visualize_3d_obj(temp, epoch, id, set_lim=True, lim=[70,70,70],save=False,
                     dir='/home/sanaz/Ryerson/Projects/TumorGAN/GAN_3D/3D_chair_results/'):
    points = get_points_from_matrix(temp, 1)
    ax = plt.axes(projection='3d')
    if len(points) > 0:
        ax.scatter3D(points[:, 0], points[:, 1], points[:, 2], color='red')
    if set_lim:
        ax.set_xlim(0, lim[0])
        ax.set_ylim(0, lim[1])
        ax.set_zlim(0, lim[2])
        if save:

            plt.savefig(dir + str(epoch) + '_' + str(id) + '.png')
        else:
            plt.show()

def visualize_3d(temp, epoch=0, id=0, set_lim = False, lim=[240,240,160],conditions =[1,2],save=False,
                 dir='/home/sanaz/Ryerson/Projects/TumorGAN/GAN_Simple_3D_shape_tumor/Results/'):
    points_tumor = get_points_from_matrix(temp, condition=conditions[0])
    points_OAR = get_points_from_matrix(temp, condition=conditions[1])
    fig = plt.figure(figsize=[10,15])
    # ax = plt.axes(projection='3d')
    flag = 0
    if len(points_tumor)>0 :
        ax1 = fig.add_subplot(1, 3, 1, projection='3d')
        ax1.scatter3D(points_tumor[:,0], points_tumor[:,1], points_tumor[:,2], color='red')
        ax2 = fig.add_subplot(1, 3, 2, projection='3d')
        ax2.scatter3D(points_tumor[:,0], points_tumor[:,1], points_tumor[:,2], color='red')

        flag = 1
    if len(points_OAR) > 0:
        ax3 = fig.add_subplot(1, 3, 3, projection='3d')
        ax3.scatter3D(points_OAR[:,0], points_OAR[:,1], points_OAR[:,2], color='blue')
        if flag == 1:
            ax1.scatter3D(points_OAR[:,0], points_OAR[:,1], points_OAR[:,2], color='blue')
        flag = 2
    if flag == 1 or flag == 2:
        if set_lim:
            if flag == 1:
                ax1.set_xlim(0,lim[0])
                ax1.set_ylim(0,lim[1])
                ax1.set_zlim(0,lim[2])

                ax2.set_xlim(0, lim[0])
                ax2.set_ylim(0, lim[1])
                ax2.set_zlim(0, lim[2])
            if flag == 2:
                ax3.set_xlim(0, lim[0])
                ax3.set_ylim(0, lim[1])
                ax3.set_zlim(0, lim[2])
        if save:

            plt.savefig(dir + str(epoch) + '_' + str(id) + '.png')
        else:
            plt.show()
def visualize_tumor_3d(xline, yline, zline, epoch=0, id=0, set_lim = False, lim=[240,240,160], color='Greens',save=False,dir='/home/sanaz/Ryerson/Projects/TumorGAN/GAN_Simple_3D_shape_tumor/Results/'):
    ax = plt.axes(projection='3d')
    ax.scatter3D(xline, yline, zline, color=color)
    # ax.plot_surface(xline, yline, zline)
    # from mpl_toolkits.mplot3d import Axes3D
    # ax = Axes3D(fig)
    # surf = ax.plot_trisurf(x, y, z, cmap=cm.jet, linewidth=0.1)
    # fig.colorbar(surf, shrink=0.5, aspect=5)
    if set_lim:
        ax.set_xlim(0,lim[0])
        ax.set_ylim(0,lim[1])
        ax.set_zlim(0,lim[2])
    if save:

        plt.savefig(dir + str(epoch) + '_' + str(id) + '.png')
    else:
        plt.show()

# def visualize_tumor_iso_gen_output(tumor, iso, gen_output, level=3):
#     tumor = tumor.reshape(240,240,160)
#     iso = iso.reshape(240,240,160)
#     gen_output = gen_output.reshape(240,240,160)
#
#     tumor_points = get_points_from_matrix(tumor, condition=1)
#     OAR_points = get_points_from_matrix(tumor, condition=2)
#
#     iso_points = get_points_from_matrix(iso, condition=1)
#
#     gen_output = edit_output_image(gen_output, level=level)
#     gen_output_points = get_points_from_matrix(gen_output, condition=1)
#     if len(OAR_points)>0:
#         OAR_cons = OAR_points.shape[0]
#     else:
#         OAR_cons = 0
#     print('real, iso, gen_output_points shape',tumor_points.shape,OAR_cons,
#           iso.shape, gen_output_points.shape)
#     if gen_output_points.shape[0]<tumor_points.shape[0] + OAR_cons:
#         fig = plt.figure()
#         ax = plt.axes(projection='3d')
#         ax.scatter3D(tumor_points[:, 0], tumor_points[:, 1], tumor_points[:, 2], cmap='b')
#         if len(OAR_points) > 0:
#             ax.scatter3D(OAR_points[:, 0], OAR_points[:, 1], OAR_points[:, 2], cmap='red')
#         plt.show()
#         visualize_tumor_3d(iso_points[:, 0], iso_points[:, 1], iso_points[:, 2], color='red')
#
#         visualize_tumor_3d(gen_output_points[:,0], gen_output_points[:,1], gen_output_points[:,2], color='Greens')
#     return
#


class JSD:
    # Jenson-Shannon divergence
    def KLD(self, p, q):
        if 0 in q:
            p = np.array(p)
            q = np.array(q)
            epsilon = 0.00001
            p = p + epsilon
            q = q + epsilon
            # raise ValueError
        return sum(_p * np.log(_p / _q) for (_p, _q) in zip(p, q) if _p != 0)

    def JSD_core(self, p, q):
        M = [0.5 * (_p + _q) for _p, _q in zip(p, q)]
        return 0.5 * self.KLD(p, M) + 0.5 * self.KLD(q, M)

def evaluate_KLD(p,q):
    '''
    :param p: probability of item
    :param q: probability of training data
    :return: KL divergence
    '''
    # dict_key = Constant_dict[key_q]
    # q = dict_key['probs']
    p = np.array(p)
    q = np.array(q)
    KL_thresh, KL_Score_thresh= JSD().KLD(p, q), JSD().JSD_core(p, q)
    # print('KL thresh', KL_thresh)
    # print('KL_Score thresh', KL_Score_thresh)
    return KL_thresh, KL_Score_thresh

def get_prob(Sample,key_q):
    dict_key = Constant_dict[key_q]
    dict_keys = list(dict_key.keys())
    if 'interval' in dict_keys:
        intervals = dict_key['interval']
        min_value = min(np.min(Sample), np.min(intervals))
        max_value = max(np.max(Sample), np.max(intervals))
        # if np.min(intervals) <= np.min(Sample)  and np.max(Sample) <= np.max(intervals):
        probs_output, bin_edges = np.histogram(Sample, bins=intervals, density=False)
        if np.sum(probs_output) == 0:
            epsilon = 0.00001
        else:
            epsilon = 0

        probs_output = probs_output / np.sum(probs_output + epsilon)
        reference_prob = dict_key['probs']


    elif 'cases' in dict_keys:
        cases = dict_key['cases']
        values = dict_key['values']
        modes1 = {}
        modes_ref = {}
        complete_set = list(set(cases+Sample))
        for i in complete_set:
            modes1[i] = 0
            modes_ref[i] = 0
        for k,i in enumerate(cases):
           modes_ref[i] += values[k]
        for i in Sample:
            if i in list(modes1.keys()):
                modes1[i] += 1
            else:
                modes1[i] = 1
        # print(modes1, cases, modes_ref)
        num_mode1 = len(modes1.keys())
        probs_output = list(modes1.values())
        probs_output = probs_output / np.sum(probs_output)

        num_modes_ref = len(modes_ref.keys())
        reference_prob = list(modes_ref.values())
        reference_prob = reference_prob / np.sum(reference_prob)
    return probs_output, reference_prob

def check_connectivity(sample,condition):
    '''
    Check connectivity of sample
    :param sample:
    :param condition:
    :return:
    '''
    flag = 0
    flag_2D = 0
    count_tum = 1
    tumor_temp = np.where(sample == condition, 1, 0)
    tumor_indice_specific = np.where(tumor_temp == 1)
    if len(np.unique(tumor_indice_specific[0])) == 1 or len(np.unique(tumor_indice_specific[1])) == 1 or len(
            np.unique(tumor_indice_specific[2])) == 1:
        # print('there is a 2D tumor here skip!')
        flag_2D = 1
        flag = 1
    if len(tumor_indice_specific[0]) == 0:
        # print('No tumor')
        count_tum = 0
        flag = 1
    if flag==0:
        tumor_temp_list = []
        s = generate_binary_structure(3,2)
        if count_tum == 1:
            labeled_array_tumor, tumor_count = label(tumor_temp,structure=s)
            # print('tumor count', tumor_count)
            for i in range(1, tumor_count + 1):
                tumor_temp_list.append(np.where(labeled_array_tumor == i, 1, 0))

        return tumor_count, tumor_temp, tumor_temp_list,count_tum, flag_2D
    else:
        return [], tumor_temp,[],count_tum, flag_2D


def extract_tumors_OARs(sample, conditions, obj='tumor'):
    sample = sample.reshape(sample.shape[1],sample.shape[2],sample.shape[3])
    tumor_temp = np.where(sample==conditions[0], 1, 0)
    OAR_temp = np.where(sample==conditions[1], 1, 0)
    flag = 0
    count_tum = 1
    count_OAR = 1
    tumor_indice_specific = np.where(tumor_temp == 1)
    if len(np.unique(tumor_indice_specific[0])) == 1 or len(np.unique(tumor_indice_specific[1])) == 1 or len(
            np.unique(tumor_indice_specific[2])) == 1:
        print('there is a 2D tumor here skip!')
        flag = 1
    if len(tumor_indice_specific[0]) == 0:
        # print('No tumor')
        count_tum = 0
        flag = 1
    tumor_indice_specific = np.where(OAR_temp == 1)
    if len(np.unique(tumor_indice_specific[0])) == 1 or len(np.unique(tumor_indice_specific[1])) == 1 or len(
            np.unique(tumor_indice_specific[2])) == 1:
        print('there is a 2D OAR here skip!')
        flag = 1
    if len(tumor_indice_specific[0]) == 0:
        # print('No OAR')
        count_OAR= 0
        flag = 1
    if flag==0:
        tumor_temp_list = []
        OAR_temp_list = []
        s = generate_binary_structure(3,2)
        if count_tum == 1:
            labeled_array_tumor, tumor_count = label(tumor_temp,structure=s)
            # print('tumor count', tumor_count)
            for i in range(1, tumor_count + 1):
                tumor_temp_list.append(np.where(labeled_array_tumor == i, 1, 0))
        if count_OAR == 1:
            labeled_array_OAR, OAR_count = label(OAR_temp,structure=s)
            # print('OAR count', OAR_count)
            for i in range(1,OAR_count+1):
                OAR_temp_list.append(np.where(labeled_array_OAR == i, 1, 0))

        return tumor_temp, OAR_temp, tumor_temp_list, OAR_temp_list
    else:
        return tumor_temp, OAR_temp,[],[]

def evaluate_one_tumor(sample, dimension, out_dir):
    flag = 0
    tumor_sizes = []
    OAR_sizes = []
    tumors_solidity_list = []
    OARs_solidity_list = []

    # tumor_indices = np.where(sample == 1)
    # OAR_indices = np.where(sample == 2)

    tumor_temp = np.where(sample==1, 1, 0)
    OAR_temp = np.where(sample==2, 1, 0)

    s = generate_binary_structure(3,2)

    labeled_array_tumor, tumor_count = label(tumor_temp,structure=s)
    print('tumor_count',tumor_count)
    for i in range(1,tumor_count+1):
        tumor_indice_specific = np.where(labeled_array_tumor==i)

        if len(np.unique(tumor_indice_specific[0])) == 1 or len(np.unique(tumor_indice_specific[1])) == 1 or len(
                np.unique(tumor_indice_specific[2])) == 1:
            print('there is a 2D tumor here skip!')
            flag = 1
            break
        tumor_temp = np.where(labeled_array_tumor == i, 1, 0)
        label_img = Label_measure(tumor_temp, connectivity=1)
        regions = regionprops(label_img)
        # points = get_points_from_matrix(labeled_array_tumor, i)
        # tumor_size = points.shape[0]
        tumor_size = np.sum(tumor_temp)
        tumor_sizes.append(tumor_size)
        print('tumorsize', i, tumor_size)

        tumors_solidity_list.append( regions[0].solidity)

    labeled_array_OAR, OAR_count = label(OAR_temp,structure=s)
    print('OAR_count',OAR_count)
    for i in range(1,OAR_count+1):
        OAR_indice_specific = np.where(labeled_array_OAR == i)
        if len(np.unique(OAR_indice_specific[0])) == 1 or len(np.unique(OAR_indice_specific[1])) == 1 or \
                len(np.unique(OAR_indice_specific[2])) == 1:
            print('there is a 2D OAR here skip!')
            flag = 1
            break


        OAR_temp = np.where(labeled_array_OAR==i, 1, 0)
        label_img = Label_measure(OAR_temp, connectivity=1)
        regions = regionprops(label_img)
        OARsize = np.sum(OAR_temp)
        OAR_sizes.append(OARsize)
        print('OARsize', i, OARsize)
        OARs_solidity_list.append(regions[0].solidity)

    return flag, tumor_count, OAR_count, tumor_sizes, OAR_sizes,\
           tumors_solidity_list, OARs_solidity_list


def read_record(Data_path, file_name):
    with open(Data_path + file_name + ".pkl") as f:
      my_list = [json.loads(line) for line in f]
    return my_list


def get_elements(list_tumors, condition=1):
    dict_points = {}
    for index, tumors in enumerate(list_tumors):
        points_1 = get_points_from_matrix(tumors, condition=condition)
        dict_points[str(index)] = points_1
    return dict_points

def get_center_tumor(tumor_points):
    '''
    Get center of an object
    :param tumor_points:
    :return:
    '''
    max_x, min_x = np.max(tumor_points[:, 0]), np.min(tumor_points[:, 0])
    max_y, min_y = np.max(tumor_points[:, 1]), np.min(tumor_points[:, 1])
    max_z, min_z = np.max(tumor_points[:, 2]), np.min(tumor_points[:, 2])
    rx = np.round((max_x - min_x) / 2)
    ry = np.round((max_y - min_y) / 2)
    rz = np.round((max_z - min_z) / 2)
    real_center = [min_x + rx, min_y + ry, min_z + rz]
    return real_center, [rx,ry,rz], [min_x, min_y, min_z], [max_x, max_y, max_z]

def get_distance_among_points(points):
    distance_to_each_other = {}
    for i in range(points.shape[0] - 1):
        temp = []
        for j in range(1, points.shape[0]):
            if i != j:
                dist = distance.euclidean(points[i], points[j])
                temp.append(dist)
        distance_to_each_other[i] = temp
    return distance_to_each_other

def get_min_distance_distribution_among_isos(tumor_points, iso_points,target_points):
    '''
    Center point iso is excluded (only compared to its own comparable instance)
    :param tumor_points:
    :param iso_points:
    :param target_points:
    :return:
    '''
    error_target_distance_list = []
    error_target_distance = 0
    distance_to_each_other = {}
    target_distances, _ = get_target_distances(tumor_points, target_points)
    real_center, [rx, ry, rz], [min_x, min_y, min_z], [max_x, max_y, max_z] = get_center_tumor(tumor_points)

    dist = []
    # This line should be excluded
    # iso_points = np.array(iso_points)
    for i in range(iso_points.shape[0]):
        dist.append(np.abs(distance.euclidean(iso_points[i], real_center)))
    center_iso_index = np.argmin(dist)
    error_target_distance_list.append(np.min(dist))

    for i in range(iso_points.shape[0] - 1):
        for j in range(1, iso_points.shape[0]):
            if i != j and i!=center_iso_index:
                dist = distance.euclidean(iso_points[i], iso_points[j])
                set_dist = []
                for k in range(len(target_distances)):
                    set_dist.append(np.abs(target_distances[k] - dist))
                error_target_distance_list.append(np.min(set_dist))
    error_target_distance = np.sum(error_target_distance_list) / iso_points.shape[0]
    return error_target_distance_list, error_target_distance


def get_target_distances(tumor_points, target_points):
    '''
    returns the target distances and target points of the data
    :param tumor_points: points of tumors
    :param target_points: number of target isocenter
    :return: Target distances and target points of the data
    '''
    real_center, radius_list, [min_x, min_y, min_z], [max_x, max_y, max_z] = get_center_tumor(tumor_points)
    [rx,ry,rz] = radius_list
    if target_points == 7:
        target_points_list = np.array([[min_x + np.round(rx / 2), real_center[1], real_center[2]],
                                     [max_x - np.round(rx / 2), real_center[1], real_center[2]],
                                     [real_center[0], min_y + np.round(ry / 2), real_center[2]],
                                     [real_center[0], max_y - np.round(ry / 2), real_center[2]],
                                     [real_center[0], real_center[1], min_z + np.round(rz / 2)],
                                     [real_center[0], real_center[1], max_z - np.round(rz / 2)],
                                     [real_center[0], real_center[1], real_center[2]]
                                     ])
        target_distances =   [(np.sqrt(radius_list[0] ** 2 + radius_list[1] ** 2) / 2),
                            (np.sqrt(radius_list[0] ** 2 + radius_list[2] ** 2) / 2),
                            (np.sqrt(radius_list[1] ** 2 + radius_list[2] ** 2) / 2),
                            radius_list[0], radius_list[1], radius_list[2]]
    elif target_points == 13:
        target_points_list = np.array(
            [[min_x + np.round(rx / 2), real_center[1], real_center[2]],
             [max_x - np.round(rx / 2), real_center[1], real_center[2]],
             [real_center[0], min_y + np.round(ry / 2), real_center[2]],
             [real_center[0], max_y - np.round(ry / 2), real_center[2]],
             [real_center[0], real_center[1], min_z + np.round(rz / 2)],
             [real_center[0], real_center[1], max_z -np.round( rz / 2)],
             [min_x, real_center[1], real_center[2]],
             [max_x, real_center[1], real_center[2]],
             [real_center[0], min_y, real_center[2]],
             [real_center[0], max_y, real_center[2]],
             [real_center[0], real_center[1], min_z],
             [real_center[0], real_center[1], max_z],
             [real_center[0], real_center[1], real_center[2]]
             ])
        target_distances = [(np.sqrt(radius_list[0] ** 2 + radius_list[1] ** 2) / 2),
                            (np.sqrt(radius_list[0] ** 2 + radius_list[2] ** 2) / 2),
                            (np.sqrt(radius_list[1] ** 2 + radius_list[2] ** 2) / 2),
                            (np.sqrt(radius_list[0] ** 2 + radius_list[1] ** 2)),
                            (np.sqrt(radius_list[0] ** 2 + radius_list[2] ** 2)),
                            (np.sqrt(radius_list[1] ** 2 + radius_list[2] ** 2)),
                            (np.sqrt((radius_list[0] / 2) ** 2 + radius_list[1] ** 2)),
                            (np.sqrt((radius_list[0] / 2) ** 2 + radius_list[2] ** 2)),
                            (np.sqrt((radius_list[0]) ** 2 + (radius_list[1] / 2) ** 2)),
                            (np.sqrt((radius_list[0]) ** 2 + (radius_list[2] / 2) ** 2)),
                            (np.sqrt((radius_list[1] / 2) ** 2 + radius_list[2] ** 2)),
                            (np.sqrt((radius_list[1]) ** 2 + (radius_list[2] / 2) ** 2)),
                            3 * radius_list[0] / 2, 3 * radius_list[1] / 2, 3 * radius_list[2] / 2,
                            radius_list[0], radius_list[1], radius_list[2],
                            2 * radius_list[0], 2 * radius_list[1], 2 * radius_list[2]]
    target_distances = np.unique(target_distances)
    return target_distances, target_points_list



def GAN_3D_iso_sphere_full_evaluation(fake_imgs, target_points, dataset_include_tumor, dimension):\
    '''
    returns a dictionary with information on the unique values of generated shape, its isocenters and their coordinates
    :param fake_imgs: input samples
    :param target_points: number of isocenters
    :param dataset_include_tumor: whether samples include tumor
    :param dimension: dimension of sample
    :return: a dictionary
    '''
    result_dict = {}
    step = 1 / (target_points + 1)
    for i in range(fake_imgs.shape[0]):
        temp = fake_imgs[i].detach().cpu().numpy().reshape(dimension[0], dimension[1], dimension[2])
        # print(np.min(temp), np.max(temp), np.unique(temp))
        # temp1 = -1 * np.ones(shape=(dimension[0], dimension[1], dimension[2]))
        temp1 = np.floor(temp * (target_points + 1))
        dict_iso = {}
        center_iso = []
        for j in range(target_points + 1):

            iso_points = get_points_from_matrix(temp1, j)
            min_coord =[]
            max_coord = []
            tumor_count= []
            if dataset_include_tumor:
                if len(iso_points)>0 and j!= 0 and j!=1:
                    min_coord = np.array([np.min(iso_points[:,0]), np.min(iso_points[:,1]),np.min(iso_points[:,2])])
                    max_coord = np.array([np.max(iso_points[:,0]), np.max(iso_points[:,1]),np.max(iso_points[:,2])])
                    center_iso.append(min_coord + np.round(0.5*(max_coord-min_coord)))
                    tumor_count, tumor_temp, tumor_temp_list,count_tum, flag_2D = check_connectivity(temp1, j)
            else:
                if len(iso_points)>0 and j!= 0:
                    min_coord = np.array([np.min(iso_points[:,0]), np.min(iso_points[:,1]),np.min(iso_points[:,2])])
                    max_coord = np.array([np.max(iso_points[:,0]), np.max(iso_points[:,1]),np.max(iso_points[:,2])])
                    center_iso.append(min_coord + np.round(0.5*(max_coord-min_coord)))
                    tumor_count, tumor_temp, tumor_temp_list,count_tum, flag_2D = check_connectivity(temp1, j)

            dict_iso[j] = {'iso_points':iso_points, 'connected':tumor_count}
        # print(np.min(temp1), np.max(temp1), np.unique(temp1))

        # print(np.min(temp1), np.max(temp1),np.unique(temp1))
        values, counts = np.unique(temp1, return_counts=True)
        # print('val',values, 'counts', counts, 'tumor_count', tumor_count)
        result_dict[i] = {'object':temp1,'min':np.min(temp1), 'max':np.max(temp1), 'unique':values, 'unique_counts':counts,'center_iso':np.array(center_iso),'dict_iso':dict_iso}
    return result_dict
