
from cvxpy import *
import numpy as np
import matplotlib.pyplot as plt
import dccp
import json
from tabulate import tabulate

from mpl_toolkits import mplot3d
import pickle
from helper import *
from scipy.spatial import distance
from discrete_frechet_master.distances.discrete import DiscreteFrechet, LinearDiscreteFrechet, VectorizedDiscreteFrechet
from discrete_frechet_master.distances.discrete import euclidean

def check_met(key, data ):
    if key in list(data.keys()):
        temp = (data[key])
    else:
        temp = 'None'
    return temp


def extract_config(path, files, keys_access):
    output_dict= {}
    my_nums = []
    output = []
    for file_name in files:
        my_nums.append(file_name)
        f = open(path + str(file_name) + '/config.json', )
        data = json.load(f)
        for key in keys_access:
            output_dict[key] = check_met(key, data )
        output.append(output_dict)

    return output


def set_values_in_array(temp, indexes, value):
    index_list = []
    for i in range(indexes.shape[1]):
        index_list.append(indexes[:,i])
    temp[index_list] = value
    return temp

def iso_detection(r_values, border_constraint,real_center=None):
    goal_ratio = []
    c_list = []
    n_values = []
    for r_value in r_values:
        n = int(np.max(border_constraint)/ (2 * r_value)) #int(np.min(border_constraint)/ (2 * r_value))
        if n <1 :
            print(n)
            continue
        n_values.append(n)
        print('r', r_value, 'n', n)
        r = [r_value for i in range(n)]
        c = Variable(shape=(n,3))
        constr = []
        for i in range(n-1):
            for j in range(i+1,n):
                a = np.array([r[i],0,0])
                a2 = np.array([0,r[i],0])
                a3 = np.array([0,0,r[i]])

                constr.append(norm(c[i,:]-c[j,:])>=r[i]+r[j])
                constr.append(norm((c[i,:]-a)-(c[i,:]+a))<=np.min(border_constraint))
                constr.append(norm((c[i,:]-a)-(c[i,:]+a))<=np.min(border_constraint))
                constr.append(norm((c[i,:]-a)-(c[i,:]+a))<=np.min(border_constraint))

        prob = Problem(Minimize(max(max(abs(c),axis=1)+r)), constr)
        #prob = Problem(Minimize(max_entries(normInf(c,axis=1)+r)), constr)
        prob.solve(method = 'dccp', ccp_times = 1)

        l = max(max(abs(c),axis=1)+r).value*2
        pi = np.pi
        ratio = pi*sum(square(r)).value/square(l).value
        print("ratio =", ratio)
        print(prob.status)
        # print(c)
        # plot
        # plt.figure(figsize=(5,5))
        fig = plt.figure()

        # circ = np.linspace(0,2*pi,num=200)
        # circ_phi  = np.linspace(0,pi,num=200)
        u, v = np.mgrid[0:2 * np.pi:30j, 0:np.pi:20j]

        x_border = [-l/2, l/2, l/2, -l/2, -l/2]
        y_border = [-l/2, -l/2, l/2, l/2, -l/2]
        z_border = [-l/2, -l/2, l/2, l/2, -l/2]
        ax = plt.axes(projection='3d')
        for i in range(n):
          x = r[i] * np.cos(u) * np.sin(v)
          y = r[i] * np.sin(u) * np.sin(v)
          z = r[i] * np.cos(v)
          ax.plot_surface(c[i,0].value+x+real_center[0], c[i,1].value+y+real_center[1], c[i,2].value+z+real_center[2], cmap=plt.cm.YlGnBu_r)
          ax.set_xlim(0, 16)
          ax.set_ylim(0, 16)
          ax.set_zlim(0, 16)
        plt.show()
        whole_vol = 0
        max_vol = border_constraint[0] * border_constraint[1] * border_constraint[2]
        for i in r:
          cube_vol = (4/3) *np.pi* (i*i*i)
          whole_vol += cube_vol
        cover_ratio = whole_vol/max_vol
        goal_ratio.append(cover_ratio)
        c_list.append(c)
        print(max_vol-whole_vol, whole_vol/max_vol)
    print('max',np.max(cover_ratio))
    print('r, n, c', r_values[np.argmax(cover_ratio)], n_values[np.argmax(cover_ratio)], c_list[np.argmax(cover_ratio)].value)
    return np.max(cover_ratio), r_values[np.argmax(cover_ratio)], n_values[np.argmax(cover_ratio)], c_list[np.argmax(cover_ratio)].value, border_constraint


def Isocenter_distribution_evaluation(isocenters, tumor, r_values_isocenters=[4,8,16]):
    # tumor largest radius
    tumor_points = get_points_from_matrix(tumor, condition=1)
    max_x, min_x = np.max(tumor_points[:,0]), np.min(tumor_points[:,0])
    max_y, min_y = np.max(tumor_points[:,1]), np.min(tumor_points[:,1])
    max_z, min_z = np.max(tumor_points[:,2]), np.min(tumor_points[:,2])
    real_center = [min_x + (max_x-min_x)/2, min_y + (max_y-min_y)/2, min_z + (max_z-min_z)/2]
    # tumor_largest_radius = np.max([max_x-min_x, max_y-min_y, max_z-min_z])
    cover_ratio,r, n, c, border_constraint = iso_detection(r_values_isocenters, [max_x-min_x, max_y-min_y, max_z-min_z],real_center)
    return cover_ratio,r, n, c, border_constraint,real_center


def iso_evaluation_tumor(file):
    with open(file, "rb") as f:
        obj = pickle.load(f)
    keys = list(obj.keys())
    vals = list(obj.values())
    len_keys = len(keys)
    count = 0
    for itr in range(len(keys)):

        if len(obj[itr]['tumor']) == 1 and obj[itr]['tumor'][0] < 900:

            count_iso_inside = 0
            # print('iso', len(obj[itr]['iso_points']))
            if len(obj[itr]['iso']) > 0:
                isocenters = np.zeros((16,16,16))
                tumor =  np.zeros((16,16,16))
                tumor = set_values_in_array(tumor, obj[itr]['tumor_points']['0'], 1)

                for j in range(len(obj[itr]['iso'])):
                    # print('iso', j, obj[itr]['iso_points'][str(j)].shape)
                    iso_obj = obj[itr]['iso_points'][str(j)]

                    for k in range(iso_obj.shape[0]):
                        if iso_obj[k,:] in obj[itr]['tumor_points']['0']:
                            count_iso_inside += 1
                    isocenters = set_values_in_array(isocenters, iso_obj, 1)

                print(count_iso_inside, count_iso_inside/np.sum(obj[itr]['iso']))
                cover_ratio, r, n, c,border_constraint,real_center = Isocenter_distribution_evaluation(isocenters, tumor, r_values_isocenters=[1, 2])
              
            count +=1
    print(count/len(keys))

def iso_evaluation_sphere(file, target_points = 7):
    all_tumor_counts = []
    all_tumor_sizes = []
    all_iso_counts = []
    all_iso_sizes = []
    iso_count = []
    tumor_size = []
    iso_size = []
    error_FD = []
    iso_all = []
    sum_iso_sizes = []
    sum_tumor_sizes = []
    with open(file, "rb") as f:
        obj = pickle.load(f)
    keys = list(obj.keys())
    vals = list(obj.values())
    len_keys = len(keys)
    print(len(keys))
    count = 0
    distances_list = []
    distance_surface_list = []
    distance_ratio_list = []
    distance_to_each_other_list = []
    error_MSE_all = []
    all_samples = []
    error_target_distance_all = []
    for itr in range(len(keys)):
        all_tumor_counts.append( len(obj[itr]['tumor']))
        sum_tumor_sizes.append(np.sum(obj[itr]['tumor']))
        all_tumor_sizes.append(obj[itr]['tumor'])
        all_iso_counts.append( len(obj[itr]['iso']))
        sum_iso_sizes.append(np.sum(obj[itr]['iso']))
        all_iso_sizes.append(obj[itr]['iso'])

        if len(obj[itr]['tumor']) == 1 and obj[itr]['tumor'][0] < 900:
            # print('itr',itr)
            count_iso_inside = 0
            # print('iso', len(obj[itr]['iso_points']))
            if len(obj[itr]['iso']) > 0:
                tumor_points = obj[itr]['tumor_points']['0']
                real_center, [rx, ry, rz], [min_x, min_y, min_z], [max_x, max_y, max_z] = get_center_tumor(tumor_points)

                isocenters = []
                for j in range(len(obj[itr]['iso'])):
                    if j ==0:
                        isocenters = obj[itr]['iso_points'][str(j)]
                    else:
                        isocenters = np.concatenate((isocenters, obj[itr]['iso_points'][str(j)]),axis=0)
                    # print('iso', j, obj[itr]['iso_points'][str(j)].shape)
                    iso_obj = obj[itr]['iso_points'][str(j)]

                    for k in range(iso_obj.shape[0]):
                        if iso_obj[k,0] <=max_x and min_x <=iso_obj[k,0]:
                            if iso_obj[k,1] <=max_y and min_y <=iso_obj[k,1]:
                                if iso_obj[k,2] <=max_z and min_z <=iso_obj[k,2]:

                        # if (iso_obj[k,:] == obj[itr]['tumor_points']['0']).all(1).any():
                        # if iso_obj[k,:] in obj[itr]['tumor_points']['0']:
                                    count_iso_inside += 1
                    # isocenters = set_values_in_array(isocenters, iso_obj, 1)
                # print('tumor', np.sum(obj[itr]['tumor']))
                error_target_distance, error_MSE, distances, distance_surface, distance_ratio, distance_to_each_other = iso_sphere_distance_evaluation(tumor_points, isocenters, real_center, [rx,ry,rz], target_points, center_flag=False)
                error_target_distance_all.append(error_target_distance)
                distances_list.append(distances)
                distance_surface_list.append(distance_surface)
                distance_ratio_list.append(distance_ratio)
                distance_to_each_other_list.append(distance_to_each_other)
                iso_size.append(np.sum(obj[itr]['iso']))
                iso_count.append(len(obj[itr]['iso']))
                tumor_size.append(np.sum(obj[itr]['tumor']))
                iso_all.append(count_iso_inside/np.sum(obj[itr]['iso']))
                error_MSE_all.append(error_MSE)

                _, points = get_target_distances(tumor_points, target_points)
                frechet0 = DiscreteFrechet(euclidean)
                dis = frechet0.distance(points, iso_obj)
                error_FD.append(dis)
                samples = square_distribution(iso_obj, [max_x,max_y,max_z], [min_x, min_y, min_z], 3)
                all_samples.append(samples)

            count +=1
    # print(count/len(keys))
    tumor_coverage = count/len(keys)
    arguments = [error_target_distance_all, all_samples, iso_count, tumor_size, iso_size, error_FD, tumor_coverage, iso_all, all_tumor_counts ,\
            all_tumor_sizes,all_iso_counts,all_iso_sizes, sum_tumor_sizes, sum_iso_sizes,distances_list,\
            distance_surface_list, distance_ratio_list, distance_to_each_other_list, error_MSE_all]
    names = ['error_target_distance_all', 'all_samples', 'iso_count', 'tumor_size', 'iso_size', 'error_FD', 'tumor_coverage', 'iso_all', 'all_tumor_counts' ,\
            'all_tumor_sizes','all_iso_counts','all_iso_sizes', 'sum_tumor_sizes', 'sum_iso_sizes','distances_list',\
            'distance_surface_list', 'distance_ratio_list', 'distance_to_each_other_list', 'error_MSE_all']
    result_return = set_values_in_dict(arguments, names)

    return  result_return

def iso_evaluation_sphere_iso_filled(file, dataset_include_tumor, target_points=7):
    result_return = {}
    all_tumor_counts = []
    all_tumor_sizes = []
    all_iso_counts = []
    all_iso_sizes = []
    iso_count = []
    tumor_size = []
    iso_size = []
    iso_all = []
    sum_iso_sizes = []
    sum_tumor_sizes = []
    with open(file, "rb") as f:
        obj = pickle.load(f)
    keys = list(obj.keys())
    vals = list(obj.values())
    len_keys = len(keys)
    count = 0
    distances_list = []
    distance_surface_list = []
    distance_ratio_list = []
    distance_to_each_other_list = []
    error_MSE_all = []
    all_samples = []
    error_target_distance_all = []
    keys = list(obj.keys())
    connected_objects = []
    unique_all = []
    unique_counts_all = []
    iso_points_all = []
    error_target_distance_all = []
    error_FD = []
    count = 0
    all_tumor_sizes = []
    error_count_isos_all = []
    count_connected_tumor = 0
    iso_target_equal = 0
    True_count = 0
    iso_list= []
    for key in keys:
        # print(key)
        temp = obj[key]['object']
        # visualize_multi_colored_isos(temp, 'iso',0, 0, [], [16,16,16], set_lim=True, save=False)

        tumor = np.where(temp != 0, 1, 0)
        tumor_points = get_points_from_matrix(tumor, condition=1)
        # ax = plt.axes(projection='3d')
        # ax.scatter3D(tumor_points[:, 0], tumor_points[:, 1], tumor_points[:, 2], color='red')
        # plt.show()
        iso_points = obj[key]['center_iso']
        iso_points = np.array(iso_points)
        iso_list.append(iso_points)
        # print(key, obj[key]['unique'], len(tumor_points),  iso_points.shape[0])

        iso_points = iso_points.reshape(iso_points.shape[0], 3)
        iso_count.append(iso_points.shape[0])

        error_count_iso = np.abs(target_points-iso_points.shape[0])
        error_count_isos_all.append(error_count_iso)

        unique =  obj[key]['unique']
        unique_counts =  obj[key]['unique_counts']
        # print(unique, unique_counts)
        if iso_points.shape[0] == target_points:
            iso_target_equal+=1
        else:
            continue
        if len(tumor_points) ==0 or iso_points.shape[0] == 0:
            continue
        else:
            count += 1
        # min =  obj[key]['min']
        # max =  obj[key]['max']



        unique_all.append(unique)
        unique_counts_all.append(unique_counts)
        dict_iso = obj[key]['dict_iso']
        connected_isos = []
        points_isos = []
        # is tumor connected?

        tumor_count, tumor_temp, tumor_temp_list, count_tum, flag_2D = check_connectivity(tumor, 1)
        if tumor_count == 1:
            count_connected_tumor +=1
        all_tumor_counts.append(tumor_count)
        sum_tumor_sizes.append(tumor_points.shape[0])

        real_center, [rx, ry, rz], [min_x, min_y, min_z], [max_x, max_y, max_z] = get_center_tumor(tumor_points)
        if target_points >1:
            error_target_distance, error_MSE, distances_to_center, distance_surface, distance_ratio, distance_to_each_other = iso_sphere_distance_evaluation(
                tumor_points, iso_points, real_center, [rx, ry, rz], target_points, center_flag=False)

            error_target_distance_all.append(error_target_distance)
            distances_list.append(distances_to_center)
            distance_surface_list.append(distance_surface)
            distance_ratio_list.append(distance_ratio)
            distance_to_each_other_list.append(distance_to_each_other)
            error_MSE_all.append(error_MSE)

            _, points = get_target_distances(tumor_points, target_points)
            points = points.astype(int)
            # iso_points = iso_points.astype(int)
            frechet0 = DiscreteFrechet(euclidean)
            dis = frechet0.distance(points, iso_points)
            error_FD.append(dis)
            #
        if dataset_include_tumor:
            start_index = 2
        else:
            start_index = 1
        for dict_iso_key in range(start_index,len(list(dict_iso.keys()))):
            iso_points_key = dict_iso[dict_iso_key]['iso_points']
            # min_coords = dict_iso[dict_iso_key]['min_coord']
            # max_coord = dict_iso[dict_iso_key]['max_coord']
            connected = dict_iso[dict_iso_key]['connected']
            # object_points = dict_iso[dict_iso_key]['objects_points']
            connected_isos.append(connected)
            points_isos.append(iso_points_key)

        connected_objects.append(connected_isos)

    connected_value_unique = []
    connected_count_unique = []
    # connected_value_unique, connected_count_unique = list_unique(connected_objects)

    coverage_tumors = iso_target_equal / (len(keys))
    connected_tumor = count_connected_tumor / (len(keys))
    if count > 0:
        arguments = [ connected_value_unique,connected_count_unique, coverage_tumors, connected_tumor, unique_all, unique_counts_all,
                      error_target_distance_all,error_FD,\
               all_tumor_counts ,iso_points_all,iso_count, \
                sum_tumor_sizes, connected_objects, distances_list, distance_surface_list,\
                distance_ratio_list, distance_to_each_other_list,error_MSE_all,connected_objects,iso_list, error_count_isos_all]
        names = ['connected_value_unique','connected_count_unique', 'coverage_tumors', 'connected_tumor', 'unique_all', 'unique_counts_all', 'error_target_distance_all','error_FD',\
               'all_tumor_counts' , 'iso_points_all','iso_count',\
                'sum_tumor_sizes', 'connected_objects', 'distances_list', 'distance_surface_list',\
                'distance_ratio_list', 'distance_to_each_other_list','error_MSE_all','connected_objects','iso_list', 'error_count_isos_all']
        result_return = set_values_in_dict(arguments, names)
    return result_return

def find_point_on_perimeter(points):
    a_temp = np.array([[1,0,0],[0,1,0],[0,0,1],
                       [-1,0,0],[0,-1,0],[0,0,-1]])
    primeter_temp = []
    flag = 0
    for i in range(len(points)):
        flag = 0
        for j in range(len(a_temp)):
            if flag == 0:
                a = points[i] + a_temp[j]
                if not (points==a).all(1).any():
                    primeter_temp.append(points[i])
                    flag = 1

    return primeter_temp

def check_points_inside_cube(coordinates_max, coordinates_min, point):
    flag = 0
    for j in range(len(point)):
        if coordinates_min[j] > point[j] or point[j] > coordinates_max[j]:
            flag = 1
    if flag == 0:
        return True
    else:
        return False

def square_distribution(isocenter_points, coordinates_max, coordinates_min, num_cubes, dim=3):
    coordinates_max_list = []
    coordinates_min_list = []
    widthx = (coordinates_max[0] - coordinates_min[0]) / num_cubes
    widthy = (coordinates_max[1] - coordinates_min[1]) / num_cubes
    widthz = (coordinates_max[2] - coordinates_min[2]) / num_cubes
    for i in range(num_cubes):
        for k in range(num_cubes):
            for t in range(num_cubes):
                temp_max = [np.round(coordinates_min[0] + (i+1) * widthx,3), np.round(coordinates_min[1] + (k+1) * widthy,3), np.round(coordinates_min[2] + (t+1) * widthz,3)]
                temp_min = [np.round(coordinates_min[0] + i * widthx,3), np.round(coordinates_min[1] + k * widthy,3), np.round(coordinates_min[2] + t * widthz,3)]
                coordinates_max_list.append(temp_max)
                coordinates_min_list.append(temp_min)

    sample_list = []
    for i in range(len(coordinates_max_list)):
        sample = 0
        for point in isocenter_points:
            # print(coordinates_max_list[i], coordinates_min_list[i], point)
            if check_points_inside_cube(coordinates_max_list[i], coordinates_min_list[i], point):
                sample += 1
        sample_list.append(sample)
    return sample_list

def iso_sphere_distance_evaluation(tumor_points, points, center, radius_list, target_points, center_flag=False):
    distances_to_center = []
    epsilon = 0.0001
    error_target_distance_list, error_target_distance = get_min_distance_distribution_among_isos(tumor_points, points, target_points)
    distance_to_each_other = get_distance_among_points(points)
    for i in range(points.shape[0]):
        distances_to_center.append(distance.euclidean(points[i], center) + epsilon)
    if center_flag:
        index = np.argmin(distances)
        center_iso = points[np.argmin(distances)]
        distances_to_center = []
        for i in range(points.shape[0]):
            if i != index:
                distances_to_center.append(distance.euclidean(points[i], center_iso) + epsilon)


    primeter_points = find_point_on_perimeter(tumor_points)
    primeter_points = np.array(primeter_points)

    distance_surface = []
    for i in range(points.shape[0]):
        min_dist = np.min(distance.cdist(points[i].reshape(1,3),primeter_points))
        distance_surface.append(min_dist + epsilon)
    distance_ratio = []
    for i in range(len(distances_to_center)):
        distance_ratio.append(distance_surface[i]/(distances_to_center[i]+distance_surface[i]))
    error = 0
    for i in range(points.shape[0]):
        if np.abs(distance_ratio[i]) < 0.1:
            error += np.abs(distance_ratio[i])
        elif np.abs(distance_ratio[i]) > 0.1 and np.abs(distance_ratio[i]) < 0.7:
            error += np.abs(0.5-distance_ratio[i])
        elif np.abs(distance_ratio[i]) > 0.7:
            error += np.abs(1 - distance_ratio[i])
    error = error / points.shape[0]

    return error_target_distance, error, distances_to_center, distance_surface, distance_ratio, distance_to_each_other

def eval_files(files, dir, iterations_list=[], step_report=50):

    keys_access = ['GAN_model', 'filled', 'itr_critic','num_epochs','noise_dim','learning_rate_g' ,'learning_rate_d','target_points',
                   'dataset_size','shape','space_dim','batch_size','in_channels_dim', 'dimension',
                  'coef', 'learning_rate']
    for dir_file in files:
        print('file', dir_file)
        results = extract_config(dir, [dir_file], keys_access)
        print(results)
        target_points = results[0]['target_points']
        epochs = results[0]['num_epochs']
        # for i in range(13):
        iter = int(epochs/step_report)
        iterations = list(range(iter)) if len(iterations_list) == 0 else iterations_list
        for i in iterations:
            k = step_report * i - 1 if i == iter-1 else step_report*i
            file = dir + str(dir_file) + '/' +   'result_dict_epoch'+str(k)+'.pkl'
            print('file',file)
            return_results = iso_evaluation_sphere(dir + str(dir_file)+ '/result_dict_epoch'+str(k)+'.pkl', target_points)
            print_results(return_results)
            a,b  = np.unique(return_results['unique_all'],return_count=True)
            print(a, b)
    return


def get_radius(fake_points):
  center = np.median(fake_points, axis=0)
  dist = center - fake_points
  dist_all = np.sqrt(np.sum(dist**2, axis=1))
  return dist_all

def get_percentage(a):
  q75,q25 = np.percentile(a,[75,25])
  intr_qr = q75-q25
  max = q75+(1.5*intr_qr)
  min = q25-(1.5*intr_qr)
  outliers = np.where(a>max)
  return q25, q75, max, min,outliers
from scipy import stats
def create_hist(temp, xx_lim, y_lim, x_axis, y_axis, y_lim_set = False, x_lim_set=False, spinevis = True, Freq=True, kde=False, save=False, name=''):

  fig,ax = plt.subplots(figsize=(6,4))
  if Freq:
        ax.hist(temp, bins=np.linspace(0,xx_lim,100), color='lightgreen', edgecolor='green')

  else:
        xx = np.linspace(0,xx_lim,100)
        kde = stats.gaussian_kde(temp)
        ax.hist(temp, bins=100, density=True, color='lightgreen', edgecolor='green')
        ax.plot(xx, kde(xx), color='green')
  ax.set_xlabel(x_axis)
  ax.set_ylabel(y_axis)
  if y_lim_set:
    ax.set_ylim(0,y_lim)
  if x_lim_set:
    ax.set_xlim(0, xx_lim)



  ax.axvline(np.mean(temp), alpha=0.6, ymax=y_lim, color='b')
  ax.tick_params(left=False, bottom=False)
  if spinevis ==False:
    for ax, spine in ax.spines.items():
      spine.set_visible(False)
  if save:
      fig.savefig(name , bbox_inches='tight')
  else:
    plt.show()


def get_ratio(temp, points, val, thresh=None):
    a = get_radius(points)
    q25, q75, max, min, outliers = get_percentage(a)
    # print(q25, q75, max, min)

    if thresh != None:
        fake_new = points[np.where(a < thresh)]
    else:
        fake_new = points[np.where(a < q75)]

    s = generate_binary_structure(3, 2)
    _, tumor_count = label(temp, structure=s)
    temp_new = np.zeros((temp.shape[0], temp.shape[1], temp.shape[2]))
    for index, (i, j, k) in enumerate(fake_new):
        temp_new[i, k, j] = val
    _, tumor_count_new = label(temp_new, structure=s)
    flag_2D = 0
    # print(np.unique(fake_new[:, 0]), np.unique(fake_new[:, 1]), np.unique(fake_new[:, 2]))
    if len(np.unique(fake_new[:, 0])) == 1 or len(np.unique(fake_new[:, 1])) == 1 or len(np.unique(fake_new[:, 2])) == 1:
        flag_2D = 1
        print('2D')
    if flag_2D == 0:
        try:
            hull = ConvexHull(fake_new)
            hull_volume = hull.volume
            hull_ratio =fake_new.shape[0] / hull_volume
        except:
            print("An exception occurred")
            hull_ratio = None
    else:
        hull_volume = 1
        hull_ratio =fake_new.shape[0] / hull_volume
    return fake_new, outliers, hull_ratio , fake_new.shape[0] / points.shape[
        0], tumor_count, tumor_count_new, temp_new

def eval_files_sphere_filled(files, dir, name_save, iterations_list=[], step_report=50,target_points=7, dataset_include_tumor=False, save=False):

    keys_access = ['GAN_model', 'filled','size_wanted', 'itr_critic','num_epochs','generator_activation',\
    'noise_dim','learning_rate_g' ,'learning_rate_d','target_points',\
                   'dataset_size','dataset_kind','shape','space_dim','batch_size',\
                   'in_channels_dim', 'dimension',
                  'coef', 'learning_rate']
    keys_print = ['coverage_tumors', 'connected_tumor', 'error_target_distance_all',
                  'error_FD', 'iso_count', 'sum_tumor_sizes', 'error_MSE_all', 'error_count_isos_all']
    # count = len(files) *
    return_dataframe = pd.DataFrame()

    dir_files = []
    iteration_files = []
    # count = 0
    # keys_mean = []
    # keys_std = []
    for dir_file in files:
        print('file', dir_file)
        results = extract_config(dir, [dir_file], keys_access)
        print(results)
        target_points = results[0]['target_points']
        epochs = results[0]['num_epochs']
        # for i in range(13):
        iter = int(epochs/step_report)
        iterations = list(range(iter+1)) if len(iterations_list) == 0 else iterations_list

        for i in iterations:
            dir_files.append(dir_file)
            k = step_report * i - 1 if i == iter else step_report*i
            file = dir + str(dir_file) + '/' +   'result_dict_epoch'+str(k)+'.pkl'
            iteration_files.append(k)

            print('file',file)
            if os.path.isfile(file):
                return_results = iso_evaluation_sphere_iso_filled(file, dataset_include_tumor, target_points=target_points)
                # print(len(return_results.keys()))
                return_dict = print_results(return_results,keys_print)
                return_dict['file'] = dir_file
                return_dict['epoch'] = k
                return_dataframe = return_dataframe.append(return_dict,ignore_index=True)
                # print(tabulate(return_dataframe, headers='keys', tablefmt='pretty'))

            else:
                print('file is not available')
        print(tabulate(return_dataframe, headers='keys', tablefmt='pretty'))

    if save:
        return_dataframe.to_csv(dir + 'pd_result_'+name_save+'.csv')
    # print(tabulate(return_dataframe, headers='keys', tablefmt='pretty'))

    return
import matplotlib.ticker as ticker

def plot_multiple_shapes(name, temps_new, s1, s2, list_ratio_new,
                                             list_convex_new,list_points_new_size, voxel=False, trisurf=False):

    fig = plt.figure()
    index = 0
    for i1 in range(s1):
        for i2 in range(s2):
       
            ax = fig.add_subplot(s1, s2, index+1, projection='3d')
            # ax.set_axis_off()
            if voxel:
                ax.voxels(temps_new[index], edgecolor='k')
            elif trisurf:
                points = get_points_from_matrix(temps_new[index],1)
                hull = ConvexHull(points)
                s = ax.plot_trisurf(points[:, 0], points[:, 1], points[:, 2], triangles=hull.simplices,
                                    cmap='viridis', alpha=0.2, edgecolor='k')
            ax.xaxis.set_major_locator(ticker.NullLocator())
            ax.yaxis.set_major_locator(ticker.NullLocator())
            ax.zaxis.set_major_locator(ticker.NullLocator())
            ax.set_title('ratio ' + str(np.round(list_ratio_new[index],2)) +
                                            ' convex ratio ' + str(np.round(list_convex_new[index],2)) +
                                        ' point size ' + str(np.round(list_points_new_size[index],2)), fontsize=5)
            index += 1
      
    fig.savefig(name + '.pdf', bbox_inches='tight')

def check_files_convexity(file, name_data='sphere', id_name='',obj=None,save_omega=False):
    '''
    Report the evaulation metrics for connected 3D volumes 
    :param file: path of saved file
    :param name_data:  Either 'sphere' for 3D connected volume or 'Matlab' for 3D connected tumors
    :param id_name: 
    :param obj: When no file is available check this dictionary
    :param save_omega: Save the moment invariants' figures
    :return: 
    '''
    
    if obj==None:
        with open(file, "rb") as f:
            obj = pickle.load(f)
    keys = list(obj.keys())
    vals = list(obj.values())
    len_keys = len(keys)
    ratios = []
    points_size = []
    points_size_new = []
    ratio_convex_list = []
    tumor_count_list = []
    tumor_count_new_list = []
    count = 0
    moment_invariants_list = []
    moment_invariants_list_new = []
    list_temp_new = []
    list_ratio_new = []
    list_convex_new = []
    list_points_new_size = []
    tumor_sizes= []
    tumor_sizes_new = []
    flag = 0
    for key in range(len(keys)):

        temp = obj[key]['object']
        # visualize_multi_colored_isos(temp, 'iso',0, 0, [], [16,16,16], set_lim=True, save=False)

        tumor = np.where(temp != 0, 1, 0)
        tumor_points = get_points_from_matrix(tumor, condition=1)

        # print(len(tumor_points))
        if len(tumor_points) > 10:
            flag_2D = 0
            # print(np.unique(tumor_points[:, 0]), np.unique(tumor_points[:, 1]),np.unique(tumor_points[:, 2]))
            if len(np.unique(tumor_points[:, 0])) == 1 or len(np.unique(tumor_points[:, 1])) == 1 or len(
                    np.unique(tumor_points[:, 2])) == 1:
                flag_2D = 1
                print('2D')
            if flag_2D == 0:

                new_points, outliers, ratio_convex, ratio, tumor_count, tumor_count_new, temp_new = get_ratio(tumor,
                                                                                                              tumor_points,
                                                                                                              val=1)
                # tumor_sizes.append(len(tumor_points))
                # tumor_sizes_new.append(len(new_points))
                if ratio_convex != None:
                    if name_data == 'sphere':
                        if ratio > 0.6 and ratio_convex > 0.9 and tumor_count_new < 3:
                        # if ratio > 0.6 and ratio_convex > 0.9 and tumor_count_new < 2:

                            count += 1
                    elif name_data == 'Matlab':
                        if ratio > 0.6 and ratio_convex > 0.6 and tumor_count_new < 3:
                            # if ratio > 0.6 and ratio_convex > 0.9 and tumor_count_new < 2:

                            count += 1

                    ratio_convex_list.append(ratio_convex)
                ratios.append(ratio)
                points_size.append(tumor_points.shape[0])
                points_size_new.append(new_points.shape[0])
                tumor_count_list.append(tumor_count)
                tumor_count_new_list.append(tumor_count_new)
                moment_invariants_list.append(moment_invariants(tumor_points))
                moment_invariants_list_new.append(moment_invariants(new_points))

    return_results = {}
    print(count / len(keys))
    return_results['correct_spheres_percentage']  = count / len(keys)

    if name_data == 'sphere':
        keys_list = ['dataset_sphere_ellipsoid_1_40000_ratio_hist', 'dataset_sphere_ellipsoid_1_40000_convex_ratio',
                'dataset_sphere_ellipsoid_1_40000_point_size', 'dataset_sphere_ellipsoid_1_40000_point_size_new']
    elif name_data == 'Matlab':
        keys_list = ['dataset_Matlab_tumor_1_40000_ratio_hist', 'dataset_Matlab_tumor_1_40000_ratio_convex_hist',
                'dataset_Matlab_tumor_1_40000_points_hist', 'dataset_Matlab_tumor_1_40000_points_new_hist']
    print(len(ratios))
    p, q = get_prob(ratios, keys_list[0])
    KLD_ratio, KLD_score = evaluate_KLD(p, q)
    print('ratios', KLD_ratio, KLD_score)
    return_results['KLD_ratio'] = KLD_ratio

    p, q = get_prob(points_size, keys_list[2])
    KLD_points, KLD_score = evaluate_KLD(p, q)
    print('points', KLD_points, KLD_score)
    return_results['KLD_points'] = KLD_points


    p, q = get_prob(points_size_new, keys_list[3])
    KLD_points_new, KLD_score = evaluate_KLD(p, q)
    print('point_new', KLD_points_new, KLD_score)
    return_results['KLD_points_new'] = KLD_points_new

    p, q = get_prob(ratio_convex_list, keys_list[1])
    KLD_ratio_convex, KLD_score = evaluate_KLD(p, q)
    print('ratio_convex_list', KLD_ratio_convex, KLD_score)
    return_results['KLD_ratio_convex'] = KLD_ratio_convex

    dir = '/home/sanaz/Ryerson/Projects/tumorGAN/GAN_simple_3D/results/'
    moment_invariants_list = np.array(moment_invariants_list)
    print('OMEGA1',np.mean(moment_invariants_list[:,0]), np.std(moment_invariants_list[:,0]))
    return_results['OMEGA1_mean'] = np.mean(moment_invariants_list[:,0])
    return_results['OMEGA1_std'] = np.std(moment_invariants_list[:,0])

    print('OMEGA2',np.mean(moment_invariants_list[:,1]), np.std(moment_invariants_list[:,1]))
    return_results['OMEGA2_mean'] = np.mean(moment_invariants_list[:,1])
    return_results['OMEGA2_std'] = np.std(moment_invariants_list[:,1])

    print('OMEGA3',np.mean(moment_invariants_list[:,2]), np.std(moment_invariants_list[:,2]))
    return_results['OMEGA3_mean'] = np.mean(moment_invariants_list[:,2])
    return_results['OMEGA3_std'] = np.std(moment_invariants_list[:,2])

    print('point size',np.mean(points_size), np.std(points_size) )
    print('point new size',np.mean(points_size_new), np.std(points_size_new) )
    return_results['points_size_mean'] = np.mean(points_size)
    return_results['points_size_std'] = np.std(points_size)

    return_results['points_size_new_mean'] = np.mean(points_size_new)
    return_results['points_size_new_std'] = np.std(points_size_new)

    if save_omega:
        count, val = np.histogram(moment_invariants_list[:, 0], bins=100)
        create_hist(moment_invariants_list[:, 0], 2,2000 ,  'OMEGA 1', 'Frequency', y_lim_set=True, x_lim_set=True, spinevis=True, Freq=True, kde=False,
                    save=True,
                    name= dir + name_data + '_'+id_name+'_OMEGA1_Frequency_with_axis.pdf')
        count, val = np.histogram(moment_invariants_list[:, 1], bins=100)
        create_hist(moment_invariants_list[:, 1], 10,2000  ,'OMEGA 2', 'Frequency',y_lim_set=True, x_lim_set=True, spinevis=True, Freq=True, kde=False,
                    save=True,
                    name=dir + name_data + '_'+id_name+'_OMEGA2_Frequency_with_axis.pdf')
        count, val = np.histogram(moment_invariants_list[:, 2], bins=100)
        create_hist(moment_invariants_list[:,2], 120, 2000  , 'OMEGA 3', 'Frequency', y_lim_set=True, x_lim_set=True,spinevis=True, Freq=True, kde=False,
                    save=True,
                    name=dir + name_data + '_'+id_name+'_OMEGA3_Frequency_with_axis.pdf')

        create_hist(moment_invariants_list[:, 0], 2,2000  ,  'OMEGA 1', 'Frequency', y_lim_set=True, x_lim_set=True, spinevis=False, Freq=True, kde=False,
                    save=True,
                    name= dir + name_data + '_'+id_name+'_OMEGA1_Frequency_without_axis.pdf')
        count, val = np.histogram(moment_invariants_list[:, 1], bins=100)
        create_hist(moment_invariants_list[:, 1], 10,2000  ,'OMEGA 2', 'Frequency', y_lim_set=True, x_lim_set=True, spinevis=False, Freq=True, kde=False,
                    save=True,
                    name=dir + name_data + '_'+id_name+'_OMEGA2_Frequency_without_axis.pdf')
        count, val = np.histogram(moment_invariants_list[:, 2], bins=100)
        create_hist(moment_invariants_list[:,2], 120,2000  , 'OMEGA 3', 'Frequency', y_lim_set=True, x_lim_set=True,spinevis=False, Freq=True, kde=False,
                    save=True,
                    name=dir + name_data + '_'+id_name+'_OMEGA3_Frequency_without_axis.pdf')
   
    return return_results

def check_files_sphere_filled_iso(file_list,dataset_include_tumor=False, name_data='sphere',data_path='/home/sanaz/Ryerson/Projects/tumorGAN/GAN_simple_3D/results/'):
    '''
    Evaluate the connected volumes packed with spheres
    :param file_list: location of logged dictionary files
    :param dataset_include_tumor: whether the data include tumor
    :param name_data: Either 'sphere' for 3D connected volume or 'Matlab' for 3D connected tumors
    :param data_path: Path of saved logs (Sacred logs outputs)
    :return: return_results (dictionary with evaluation metrics)
    '''
    val_list_files  = []
    count_list_files = []
    # file_list = [ "result_dict_epoch600.pkl"]
    for itr in range(len(file_list)):
        print(file_list[itr])
        with open(file_list[itr], "rb") as f:
            obj = pickle.load(f)
        keys = list(obj.keys())
        vals = list(obj.values())
        len_keys = len(keys)
        val_list = []
        count_list = []
        ratios_keys = []
        points_size_keys = []
        points_size_new_keys = []
        ratio_convex_list_keys = []
        tumor_count_list_keys = []
        tumor_count_new_list_keys = []
        iso_coverage_of_tumor =[]
        count_ratio = 0
        count_per_tumor_list = []
        tumor_sizes_list = []
        iso_each_cover_tumor = []
        result= []
        for key in range(len(keys)):
            ratios = []
            points_size = []
            points_size_new = []
            ratio_convex_list = []
            tumor_count_list = []
            tumor_count_new_list = []

            temp = obj[key]['object']
            val, count = np.unique(temp, return_counts=True)

            if dataset_include_tumor:
                if np.sum((val==1)) > 0:
                    tumor_sizes_list.append(count[val == 1][0])
                else:
                    tumor_sizes_list.append(0)
                val_list.append([i for i in val if i != 0 and i != 1])
                count_list.append([count[i] for i in range(len(val)) if int(val[i]) != 0 and int(val[i]) != 1])
                iso_coverage_of_tumor.append(np.sum(count_list[key])/np.sum(count[val!=0]))
                iso_each_cover_tumor.append((count_list[key]/np.sum(count[val!=0])).tolist())

            else:
                val_list.append([i for i in val if i != 0])
                count_list.append([count[i] for i in range(len(val)) if int(val[i]) != 0])
            if len(val_list[key])>1:
                # print(val_list[key])
                result.append(shannon_formula(val, count, dataset_include_tumor))

                temp1 = np.zeros((temp.shape[0], temp.shape[1], temp.shape[2]))
                count_per_tumor = 0
                for value in range(1+int(dataset_include_tumor), int(np.max(val))+1):
                    tumor = np.where(temp == value, 1, 0)
                    tumor_points = get_points_from_matrix(tumor, condition=1)

                    if len(tumor_points) > 10:
                        flag_2D = 0
                        if len(np.unique(tumor_points[:, 0])) == 1 or len(
                                np.unique(tumor_points[:, 1])) == 1 or len(
                            np.unique(tumor_points[:, 2])) == 1:
                            flag_2D = 1
                            print('2D')
                        if flag_2D == 0:

                            new_points, outliers, ratio_convex, ratio, tumor_count, tumor_count_new, temp_new = get_ratio(tumor,
                                                                                                                          tumor_points,
                                                                                                                          value)
                            if ratio_convex != None:
                                if name_data == 'sphere':
                                    if ratio > 0.6 and ratio_convex > 0.9 and tumor_count_new < 3:
                                        count_ratio += 1
                                        count_per_tumor+=1
                                elif name_data == 'Matlab':
                                    if ratio > 0.4 and ratio_convex > 0.4 and tumor_count_new < 3:
                                        count_ratio += 1
                                        count_per_tumor+=1


                                ratio_convex_list.append(np.round(ratio_convex, 2))



                            temp1 += temp_new
                            ratios.append(np.round(ratio, 2))
                            points_size.append(tumor_points.shape[0])
                            points_size_new.append(new_points.shape[0])
                            tumor_count_list.append(tumor_count)
                            tumor_count_new_list.append(tumor_count_new)
            
                count_per_tumor_list.append(count_per_tumor/len(val_list[key]))
                ratios_keys.append(ratios)
                points_size_keys.append(points_size)
                points_size_new_keys.append(points_size_new)
                ratio_convex_list_keys.append(ratio_convex_list)
                tumor_count_list_keys.append(tumor_count_list)
                tumor_count_new_list_keys.append(tumor_count_new_list)
     
        return_results = {}
        if name_data == 'sphere':
            t_count = 0
            for index, v in enumerate(val_list):
                if len(v) == 7:
                    t_count += 1

            print('acc 7', t_count / len(keys))
            return_results['acc_with_target_unique'] = t_count / len(keys)
        if dataset_include_tumor:
            print('iso_coverage_tumor', np.mean(iso_coverage_of_tumor), np.std(iso_coverage_of_tumor))
            print('iso_each_cover_tumor', np.mean([i for j in iso_each_cover_tumor for i in j]), np.std([i for j in iso_each_cover_tumor for i in j]))

        print('diversity', np.mean(result), np.std(result))
        # print('count_ratio', count_ratio/len(keys))
        print('count_ratio', np.sum(np.array(count_per_tumor_list)>=0.15))

        print('count_per_tumor_list_mean', np.mean(count_per_tumor_list), np.std(count_per_tumor_list))
        print('tumor sizes', np.mean(tumor_sizes_list), np.std(tumor_sizes_list))
        print('Iso sizes', np.mean([i for j in count_list for i in j]), np.std([i for j in count_list for i in j]))
        print('ratio',np.mean([i for j in ratios_keys for i in j]), np.std([i for j in ratios_keys for i in j]))
        print('convex ratio',np.mean([i for j in ratio_convex_list_keys for i in j]), np.std([i for j in ratio_convex_list_keys for i in j]))
        return_results['tumor sizes_mean'] = np.mean(tumor_sizes_list)
        return_results['tumor sizes_std'] = np.std(tumor_sizes_list)
        return_results['iso sizes_mean'] = np.mean([i for j in count_list for i in j])
        return_results['iso sizes_std'] = np.std([i for j in count_list for i in j])
        return_results['ratio_mean'] = np.mean([i for j in ratios_keys for i in j])
        return_results['ratio_std'] = np.std([i for j in ratios_keys for i in j])
        return_results['convex ratio_mean'] = np.mean([i for j in ratio_convex_list_keys for i in j])
        return_results['convex ratio_std'] = np.std([i for j in ratio_convex_list_keys for i in j])

        return_results['diversity_mean'] = np.mean(result)
        return_results['diversity_std'] = np.std(result)
        return_results['count_ratio'] = count_ratio
        return_results['count_per_tumor_list_mean'] = np.mean(count_per_tumor_list)
        return_results['count_per_tumor_list_std'] = np.std(count_per_tumor_list)

        if name_data == 'sphere':
            keys_list = ['dataset_sphere_ellipsoid_7_40000_ratio_hist_with_tumor',
                    'dataset_sphere_ellipsoid_7_40000_convex_ratio_with_tumor',
                    'dataset_sphere_ellipsoid_7_40000_tumor_size', 'dataset_sphere_ellipsoid_7_40000_isocenter_sizes',
                    'dataset_sphere_ellipsoid_7_40000_count_per_tumor']
        elif name_data == 'Matlab':
            keys_list = ['dataset_Matlab_7_40000_ratio_hist_with_tumor',
                    'dataset_Matlab_7_40000_convex_ratio_hist_with_tumor',
                    'dataset_Matlab_7_40000_tumor_size_hist_with_tumor',
                    'dataset_Matlab_7_40000_isocenter_sizes_hist_with_tumor',
                    'dataset_Matlab_7_40000_count_per_tumor_hist_with_tumor',
                    'dataset_Matlab_7_40000_shannon_index_hist_with_tumor']
      
        p, q = get_prob([i for j in ratios_keys for i in j], keys_list[0])
        KLD_ratio, KLD_score = evaluate_KLD(p, q)
        print('ratios KLD', KLD_ratio, KLD_score)
        return_results['KLD_ratio'] = KLD_ratio
        print(len(tumor_sizes_list))
        p, q = get_prob(tumor_sizes_list, keys_list[2])
        KLD_points, KLD_score = evaluate_KLD(p, q)
        print('tumor sizes KLD', KLD_points, KLD_score)
        return_results['KLD_points'] = KLD_points

        p, q = get_prob([i for j in count_list for i in j], keys_list[3])
        KLD_points, KLD_score = evaluate_KLD(p, q)
        print('ISO points KLD', KLD_points, KLD_score)
        return_results['KLD_ISO_points'] = KLD_points

        p, q = get_prob([i for j in ratio_convex_list_keys for i in j], keys_list[1])
        KLD_ratio_convex, KLD_score = evaluate_KLD(p, q)
        print('ratio_convex_list KLD', KLD_ratio_convex, KLD_score)
        return_results['KLD_ratio_convex'] = KLD_ratio_convex

        p, q = get_prob(count_per_tumor_list, keys_list[4])
        KLD_ratio_convex, KLD_score = evaluate_KLD(p, q)
        print('count_per_tumor_list KLD', KLD_ratio_convex, KLD_score)
        return_results['KLD_count_per_tumor_list'] = KLD_ratio_convex

        if name_data == 'Matlab':
            p, q = get_prob(result, keys_list[5])
            KLD_ratio_convex, KLD_score = evaluate_KLD(p, q)
            print('Shannon KLD', KLD_ratio_convex, KLD_score)
            return_results['KLD_Shannon'] = KLD_ratio_convex
   
    return return_results

