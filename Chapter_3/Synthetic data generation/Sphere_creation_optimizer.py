from cvxpy import *
import numpy as np
import matplotlib.pyplot as plt
import dccp
import random
from mpl_toolkits.mplot3d import Axes3D

def generate_rs_ns_ellipsie(border_constraints = [16,16,16], count_max = 50, min_ratio = 0.4, min_r=8):
    loop_val= 0
    border_constraint = np.min(border_constraints)
    max_r_values = list(range(min_r, int(border_constraint / 2)))
    r_values_list = []
    n_values_list = []

    for i in range(len(max_r_values)):
        r_values_list = r_values_list + np.repeat(max_r_values[i], int(border_constraint / max_r_values[i])).tolist()
        n_values_list.append(int(border_constraint / max_r_values[i]))

    # print(max_r_values)
    # print(r_values_list)
    # print(n_values_list)
    count = 0
    count_actual = 0
    temp = []
    max_vol =border_constraints[0] *border_constraints[1]*border_constraints[2]

    while (count < count_max  or loop_val == 1000) and count_actual < count_max:
        loop_val += 1
        random_number = random.choices(n_values_list, k=1)
        # temp1 = random.choices(r_values_list, k=random_number[0])
        temp1 = np.array(random.choices(r_values_list,k=random_number[0]*3)).reshape(random_number[0],3).tolist()
        # print(temp1)
        whole_vol = 0
        flag_temp = []
        for index, r_vals in enumerate(temp1):
            # print(index, r_vals)
            flag_temp.append(np.sum(r_vals) <= int(border_constraint))
            cube_vol = (4 / 3) * np.pi * (r_vals[0] * r_vals[1]* r_vals[2])
            whole_vol += cube_vol
        ratio = whole_vol / max_vol
        # print(random_number[0], temp1, ratio)
        if sum(flag_temp) == 3 and ratio >= min_ratio:
            count_actual += len(flag_temp)
            count += 1
            # print(count, random_number[0], temp1, whole_vol / max_vol)
            temp.append(temp1)
    return temp

def generate_rs_ns(border_constraints = [16,16,16], count_max = 50, min_ratio = 0.4):
    loop_val= 0
    border_constraint = np.min(border_constraints)
    max_r_values = list(range(2, int(border_constraint / 2)))
    r_values_list = []
    n_values_list = []

    for i in range(len(max_r_values)):
        r_values_list = r_values_list + np.repeat(max_r_values[i], int(border_constraint / max_r_values[i])).tolist()
        n_values_list.append(int(border_constraint / max_r_values[i]))

    # print(max_r_values)
    # print(r_values_list)
    # print(n_values_list)
    count = 0
    temp = []
    max_vol =border_constraints[0] *border_constraints[1]*border_constraints[2]

    while count < count_max or loop_val == 1000:
        random_number = random.choices(n_values_list, k=1)
        temp1 = random.choices(r_values_list, k=random_number[0])
        whole_vol = 0
        for i in temp1:
            cube_vol = (4 / 3) * np.pi * (i ** 3)
            whole_vol += cube_vol
        ratio = whole_vol / max_vol
        # print(random_number[0], temp1, ratio)
        if np.sum(temp1) <= int(border_constraint) and ratio >= min_ratio:
            count += 1
            # print(count, random_number[0], temp1, whole_vol / max_vol)
            temp.append(temp1)
    return temp


def print_cent(temp):
  max_list = [np.max(temp[:,0]), np.max(temp[:,1]), np.max(temp[:,2])]
  min_list = [np.min(temp[:,0]), np.min(temp[:,1]), np.min(temp[:,2])]
  center = np.array([int(min_list[0] + (max_list[0]-min_list[0])/2), int(min_list[1] + (max_list[1]-min_list[1])/2), int(min_list[2] + (max_list[2]-min_list[2])/2)])
  return max_list, min_list, center
def fill_cube_with_spheres_diff_rs_ns(border_constraints = [16,16,16], count_max = 50, min_ratio = 0.4, min_r=8):
    # temp1 = generate_rs_ns(border_constraints, count_max, min_ratio)
    temp1 = generate_rs_ns_ellipsie(border_constraints, count_max, min_ratio, min_r=min_r)
    print('radius generated', len(temp1))
    # print(temp1)
    temp = []
    center_list = []
    radius_list = []
    counter = 0
    for set_r_values in temp1:
        counter += 1
        # print('r',set_r_values, 'n', len(set_r_values))
        r = set_r_values
        n = len(set_r_values)
        c = Variable(shape=(n,3))
        constr = []
        rconst = [np.max(r_list) for r_list in set_r_values]
        for i in range(n-1):
            for j in range(i+1,n):
                a = np.array([r[i][0],0,0])
                a2 = np.array([0,r[i][1],0])
                a3 = np.array([0,0,r[i][2]])

                constr.append(norm(c[i,:]-c[j,:])>=r[i]+r[j])
                constr.append(norm((c[i,:]-a)-(c[i,:]+a))<=border_constraints[0])
                constr.append(norm((c[i,:]-a2)-(c[i,:]+a2))<=border_constraints[1])
                constr.append(norm((c[i,:]-a3)-(c[i,:]+a3))<=border_constraints[2])

        prob = Problem(Minimize(max(max(abs(c),axis=1)+rconst)), constr)
        #prob = Problem(Minimize(max_entries(normInf(c,axis=1)+r)), constr)
        prob.solve(method = 'dccp', ccp_times = 1)
        l = max(max(abs(c),axis=1)+rconst).value*2
        pi = np.pi
        ratio = pi*sum(square(rconst)).value/square(l).value
        # print("ratio =", ratio)
        # print(prob.status)
        # print(c)
        # plot
        # plt.figure(figsize=(5,5))
        # fig = plt.figure()

        # circ = np.linspace(0,2*pi,num=200)
        # circ_phi  = np.linspace(0,pi,num=200)
        u, v = np.mgrid[0:2 * np.pi:20j, 0:np.pi:20j]

        # x_border = [-l/2, l/2, l/2, -l/2, -l/2]
        # y_border = [-l/2, -l/2, l/2, l/2, -l/2]
        # z_border = [-l/2, -l/2, l/2, l/2, -l/2]
        # ax = plt.axes(projection='3d')
        temp2 = []
        temp3 = []
        temp4 = []
        for i in range(n):
          x = r[i][0] * np.cos(u) * np.sin(v)
          y = r[i][1] * np.sin(u) * np.sin(v)
          z = r[i][2] * np.cos(v)
          # x1 = x.reshape(x.shape[0],x.shape[1],1)
          # y1 = y.reshape(y.shape[0],y.shape[1],1)
          # z1 = z.reshape(z.shape[0],z.shape[1],1)

          # a = np.concatenate((x1,y1,z1),axis=2)
          # print(a.shape)
          # temp1 = [c[i,0].value+x+border_constraint, c[i,1].value+y+border_constraint, c[i,2].value+z+border_constraint]
          # print(temp1[0].shape)
          # print('c', c[i].value)
          # print((c[i,0].value+x+border_constraint).shape)
          x2 = c[i,0].value+x+border_constraints[0]/2
          y2 = c[i,1].value+y+border_constraints[1]/2
          z2 = c[i,2].value+z+border_constraints[2]/2
          x1 = x2.reshape(x2.shape[0],x2.shape[1],1)
          y1 = y2.reshape(y2.shape[0],y2.shape[1],1)
          z1 = z2.reshape(z2.shape[0],z2.shape[1],1)

          a = np.concatenate((x1,y1,z1),axis=2)
          # print(a.shape)
          a = a.reshape(a.shape[0]*a.shape[1],3)
          max_list, min_list,center = print_cent(a)
          # print(max_list, min_list,center, r[i])
          # ax.scatter3D(a[:,0],a[:,1],a[:,2])
          temp2.append([max_list, min_list,center, r[i],a])
          temp3.append(center.tolist())
          temp4.append(r[i])
          # ax.plot_surface(c[i,0].value+x+border_constraint, c[i,1].value+y+border_constraint, c[i,2].value+z+border_constraint, cmap=plt.cm.YlGnBu_r)
        center_list.append(temp3)
        radius_list.append(temp4)
        # plt.show()
        temp.append(temp2)
        # print(temp)
        whole_vol = 0
        max_vol = border_constraints[0] *border_constraints[1]*border_constraints[2]
        for i in r:
          cube_vol = (4/3) *np.pi* (i[0]*i[1]*i[2])
          whole_vol += cube_vol
        # print(abs(max_vol-whole_vol), whole_vol/max_vol)
        print(counter, ' points generated')

    return temp, center_list, radius_list

def fill_cube_with_spheres_fixed_r_fixed_n(r_values, border_constraints = [16,16,16]):

    temp = []
    for r_value in r_values:
        n = int(border_constraint/r_value)

        # print('r',r_value, 'n', n)
        r = [r_value for i in range(n)]
        c = Variable(shape=(n,3))
        constr = []
        for i in range(n-1):
            for j in range(i+1,n):
                a = np.array([r[i],0,0])
                a2 = np.array([0,r[i],0])
                a3 = np.array([0,0,r[i]])


                constr.append(norm(c[i,:]-c[j,:])>=r[i]+r[j])
                constr.append(norm((c[i,:]-a)-(c[i,:]+a))<=border_constraint)
                constr.append(norm((c[i,:]-a2)-(c[i,:]+a2))<=border_constraint)
                constr.append(norm((c[i,:]-a3)-(c[i,:]+a3))<=border_constraint)

        prob = Problem(Minimize(max(max(abs(c),axis=1)+r)), constr)
        #prob = Problem(Minimize(max_entries(normInf(c,axis=1)+r)), constr)
        prob.solve(method = 'dccp', ccp_times = 1)
        l = max(max(abs(c),axis=1)+r).value*2
        pi = np.pi
        ratio = pi*sum(square(r)).value/square(l).value
        # print("ratio =", ratio)
        # print(prob.status)
        # print(c)
        # plot
        # plt.figure(figsize=(5,5))
        # fig = plt.figure()

        # circ = np.linspace(0,2*pi,num=200)
        # circ_phi  = np.linspace(0,pi,num=200)
        u, v = np.mgrid[0:2 * np.pi:20j, 0:np.pi:20j]

        # x_border = [-l/2, l/2, l/2, -l/2, -l/2]
        # y_border = [-l/2, -l/2, l/2, l/2, -l/2]
        # z_border = [-l/2, -l/2, l/2, l/2, -l/2]
        # ax = plt.axes(projection='3d')
        temp1 = []
        for i in range(n):
          x = r[i] * np.cos(u) * np.sin(v)
          y = r[i] * np.sin(u) * np.sin(v)
          z = r[i] * np.cos(v)
          # x1 = x.reshape(x.shape[0],x.shape[1],1)
          # y1 = y.reshape(y.shape[0],y.shape[1],1)
          # z1 = z.reshape(z.shape[0],z.shape[1],1)

          # a = np.concatenate((x1,y1,z1),axis=2)
          # print(a.shape)
          # temp1 = [c[i,0].value+x+border_constraint, c[i,1].value+y+border_constraint, c[i,2].value+z+border_constraint]
          # print(temp1[0].shape)
          # print('c', c[i].value)
          # print((c[i,0].value+x+border_constraint).shape)
          x2 = c[i,0].value+x+border_constraint/2
          y2 = c[i,1].value+y+border_constraint/2
          z2 = c[i,2].value+z+border_constraint/2
          x1 = x2.reshape(x2.shape[0],x2.shape[1],1)
          y1 = y2.reshape(y2.shape[0],y2.shape[1],1)
          z1 = z2.reshape(z2.shape[0],z2.shape[1],1)

          a = np.concatenate((x1,y1,z1),axis=2)
          # print(a.shape)
          a = a.reshape(a.shape[0]*a.shape[1],3)
          max_list, min_list,center = print_cent(a)
          # print(max_list, min_list,center, r[i])
          # ax.scatter3D(a[:,0],a[:,1],a[:,2])
          temp1 = [max_list, min_list,center, r[i],a]
          # ax.plot_surface(c[i,0].value+x+border_constraint, c[i,1].value+y+border_constraint, c[i,2].value+z+border_constraint, cmap=plt.cm.YlGnBu_r)
        # plt.show()
        temp.append(temp1)
        # print(temp)
        whole_vol = 0
        max_vol = border_constraint ** 3
        for i in r:
          cube_vol = (4/3) *np.pi* (i*i*i)
          whole_vol += cube_vol
        # print(abs(max_vol-whole_vol), whole_vol/max_vol)
    return temp


def get_target_points(center_list, radius_list, target_points=7,space_dim=3):
    center_list_new = []
    radius_list_new = []
    additional_points_list = []
    center_list_new = [b for i in center_list for b in i]
    radius_list_new = [b for i in radius_list for b in i]
    for j in range(len(center_list_new)):
        additional_points_temp = []
        center = center_list_new[j]
        r = radius_list_new[j]
        if target_points == 7:
          additional_points_temp.append(center)
          for j in range(space_dim):
              change = np.zeros((1, space_dim))
              change[0, j] = 1

              additional_points_temp.append((center + int(r[j] / 2) * change).tolist()[0])
              additional_points_temp.append((center - int(r[j] / 2) * change).tolist()[0])
          # additional_points.append([[center[0,0],center[0,1],center[0,2]], [int(center[0,0]+r/2),center[0,1],center[0,2]],[int(center[0,0]-r/2),center[0,1],center[0,2]],
          #                           [center[0,0],int(center[0,1]+r/2),center[0,2]],[center[0,0],int(center[0,1]-r/2),center[0,2]],
          #                           [center[0,0],center[0,1],int(center[0,2]+r/2)],[center[0,0],center[0,1],int(center[0,2]-r/2)]])
        elif target_points == 10:
          additional_points_temp.append(center)
          for j in range(space_dim):
              change = np.zeros((1, space_dim))
              change[0, j] = 1

              for delta in range(-r[j], r[j] + 1):
                  temp = (center + int(delta) * change).tolist()[0]
                  temp1 = [int(x) for x in temp]
                  additional_points_temp.append(temp1)
        elif target_points == 13:
          additional_points_temp.append(center)
          for j in range(space_dim):
              change = np.zeros((1, space_dim))
              change[0, j] = 1

              for delta in [-r[j], -int(r[j] / 2), int(r[j] / 2), r[j]]:
                  temp = (center + int(delta) * change).tolist()[0]
                  temp1 = [int(x) for x in temp]
                  additional_points_temp.append(temp1)
        else:
          additional_points_temp.append(center)
        additional_points_list.append(additional_points_temp)

    return center_list_new, radius_list_new, additional_points_list


