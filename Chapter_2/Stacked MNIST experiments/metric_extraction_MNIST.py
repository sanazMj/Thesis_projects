import numpy as np

path = '/home/Projects/VARNETProject/Stacked_MNIST/logs/'

files = list(range(251,259))

num_modes_list= []
kl_list =[]
loss_d_list = []
loss_g_list = []
loss_v_list = []
Inception_mean_list =[]
Inception_std_list =[]

for num in files:
    num_modes = []
    kl = []
    loss_d = []
    loss_g = []
    loss_v = []
    inception_mean= []
    inception_std= []

    my_lines =[]
    # print(path + str(num) + '/cout.txt')
    f = open(path + str(num) + '/cout.txt', 'rt')
    for my_line in f:
        my_lines.append(my_line.rstrip('\n'))
    # a = f.read()
    f.close()
    for i in range(len(my_lines)):
        Number_modes_point = my_lines[i].find('Number of modes')
        Number_finish = my_lines[i].find('out')
        kl_thresh_point = my_lines[i].find('KL' )
        loss_d_point =  my_lines[i].find('loss_d' )
        loss_g_point =  my_lines[i].find('loss_g' )
        loss_v_point =  my_lines[i].find('loss_v' )
        inception_point =  my_lines[i].find('Inception score' )

        if Number_modes_point !=-1:
            print(my_lines[i][Number_modes_point + 16: Number_finish-1])
            num_modes.append(int(my_lines[i][Number_modes_point + 16: Number_finish-1]))


        if kl_thresh_point != -1:
            # print(my_lines[i][kl_thresh_point+ 3: kl_thresh_point+10 ])
            if my_lines[i][kl_thresh_point+ 2: kl_thresh_point+ 8] == '':
                kl.append(0)
            else:
                kl.append(np.round(float(my_lines[i][kl_thresh_point+ 3: kl_thresh_point+10 ]),3))

        if loss_d_point != -1:
            # print(my_lines[i][loss_d_point + 8: loss_d_point + 13])
            loss_d.append(np.round(float(my_lines[i][loss_d_point + 8: loss_d_point + 13]), 3))
        if loss_g_point != -1:
            # print(my_lines[i][loss_g_point + 8: loss_g_point + 13])
            loss_g.append(np.round(float(my_lines[i][loss_g_point +8: loss_g_point + 13]), 3))
        if loss_v_point != -1:
            # print(my_lines[i][loss_v_point + 8: loss_v_point + 13])
            loss_v.append(np.round(float(my_lines[i][loss_v_point + 8: loss_v_point + 13]), 3))
        if inception_point != -1:
            offset = 35
            if my_lines[i][inception_point + 35]=='.':
                offset = 34

            print(my_lines[i][inception_point + 16: loss_v_point + 23],' vs ',my_lines[i][inception_point + offset: loss_v_point + 42])
            inception_mean.append(np.round(float(my_lines[i][loss_v_point + 16: loss_v_point + 23]), 4))
            inception_std.append(np.round(float(my_lines[i][loss_v_point + offset: loss_v_point + 42]), 4))

    num_modes_list.append(num_modes)
    kl_list.append(kl)
    loss_d_list.append(loss_d)
    loss_g_list.append(loss_g)
    loss_v_list.append(loss_v)
    Inception_mean_list.append(inception_mean)
    Inception_std_list.append(inception_std)


print(num_modes_list)
print(kl_list)
print(loss_d_list)
print(loss_g_list)
print(loss_v_list)
print(Inception_mean_list)
print(Inception_std_list)
