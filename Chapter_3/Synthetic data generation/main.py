# For 3D connected volume generation

from Shape_creator import *

dataset_kind = 'iso_sphere_full'
dimension_num = 3
dimension = [16, 16, 16]
dataset_size = 100000
target_points = 7
shape_choice = 'sphere'
name_choice = ''
# provide information to create the random points and save them as data
save_random_points(dataset_kind, dimension_num, dimension, dataset_size, target_points, shape=shape_choice,
                       filled_vals=[0, 1], filled=True,
                       Data_path='/home/sanaz/Ryerson/Projects/tumorGAN/Data/')
# Load data
center_mat_sphere, centers1, imgs, centers_list, radius_list = get_random_points(dataset_kind, dimension_num,
                                                                                     dimension,
                                                                                     dataset_size, target_points,
                                                                                     name_choice=name_choice,
                                                                                     shape=shape_choice)


# for 3D connected tumor generation
from Create_dataset import * 



x_limit = 16
Data_path_input = '/home/sanaz/Ryerson/Projects/tumorGAN/Data/Sphere_Feb_07/Sphere_random_Rs/' # Path of generated tumor shapes using Matlab

file_names, min_indexes_tumor, max_indexes_tumor = get_tumor_coord(x_limit,Data_path_input)
create_python_dataset_from_matfiles(file_names,'Feb_07_test',min_indexes_tumor,
                                     max_indexes_tumor,size=(x_limit,x_limit,x_limit),
                                     Data_path=Data_path_input,values=[1,1])

