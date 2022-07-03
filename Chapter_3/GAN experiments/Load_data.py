import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoaderimport h5py
import json
import os
import scipy.io as sio
from helper import evaluate_one_tumor,get_points_from_matrix,extract_tumors_OARs, Matlab_trisurf, Plotly_trisurf
from scipy.ndimage.measurements import label
import plotly.graph_objects as go
import plotly.offline as off_line

import pickle
import re

def load_data_tumor(data_file,batch_size, size=0, dim=[240,240,160], excluded_index=[]):

    if size == 0:
        count_batch = len(data_file['data'])//batch_size
    else:
        count_batch = size//batch_size

    for i in range(1, count_batch+1):
        yield torch.tensor(data_file['data'][(i - 1) * batch_size:i * batch_size])


def load_data_iso(data_file, iso_dict, batch_size):
 #  my_list: itr: {isocenters, isocenter_allocation,num_tumors,num_OARs}
    count_batch = len(iso_dict)//batch_size
    my_list = list(iso_dict.values())
    for i in range(1, count_batch+1):
        Dict_isocenters = my_list[(i-1)*batch_size:i*batch_size]
        temp_data_batch = np.zeros((batch_size,240,240,160))
        # print(i)
        for j in range(len(Dict_isocenters)):
          # print(j)
          points = Dict_isocenters[j]['isocenters']
          # points = np.array(Dict_isocenters_values)
          if len(points.shape) <3:
            point1 = points.astype(int)
          else:
            point1 = points[0].astype(int)

          # print(point1)
          temp = np.zeros((240, 240, 160))
          for k in range(point1.shape[0]):
                temp_point = point1[k,:]
                # print(temp_point)
                temp[temp_point[0], temp_point[1], temp_point[2]] = 1
          temp_data_batch[j,:] = temp

        yield torch.tensor(temp_data_batch), torch.tensor(data_file['data'][(i-1)*batch_size:i*batch_size])

