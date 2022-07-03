import re
import json


path = '/home/Projects/VARNETEES/EES/Main/logs/'


def get_digit(line):
    s = [int(number) for number in re.findall(r"\d+", line)]
    return s


def extract_inline(lines, index_list):
    acc = [get_digit(lines[index_list[0]])[-2:], get_digit(lines[index_list[1]])]
    modes_unique = [get_digit(lines[index_list[2]])[-2], get_digit(lines[index_list[3]])[-2]]
    modes_generated = [get_digit(lines[index_list[2]])[-1], get_digit(lines[index_list[3]])[-1]]
    return acc, modes_unique, modes_generated


def extract_config(files):
    my_nums = []
    level_line = []
    coef = []
    slope = []
    mode_collapse = []
    model_structure =[]
    model_type =[]
    var_coef =[]
    size = [] # Pixel_Full
    pac_num = [] # PacGAN_pacnum
    Time = []
    for file_name in files:
        my_nums.append(file_name)
        f = open(path + str(file_name) + '/run.json', )
        data = json.load(f)
        temp = get_digit(data['start_time'][11:])
        temp1 = get_digit(data['stop_time'][11:])
        print(temp, temp1)
        Time.append(-(temp[0] * 3600 + temp[1] * 60 + temp[2]) +
                    (temp1[0] * 3600 + temp1[1] * 60 + temp1[2]))
    print(Time)
    for file_name in files:
        my_nums.append(file_name)
        f = open(path + str(file_name) + '/config.json', )
        data = json.load(f)
        var_coef.append(data['var_coef'])
        coef.append(data['Coef'])
        slope.append(data['Slope'])
        level_line.append(data['Level_line'])
        mode_collapse.append(data["mode_collapse"])
        model_structure.append(data["model_structure"])
        model_type.append(data["model_type"])
        size.append(data["Pixel_Full"]) # Pixel_Full
        pac_num.append(data["PacGAN_pacnum"])# PacGAN_pacnum
    print('files',my_nums)
    print('level_line =',level_line)
    print('slope=',slope)
    print('coef=', coef)
    print('var_coef=', var_coef)
    print('mode_collapse=',mode_collapse)
    print('model_structure=',model_structure)
    print('model_type=',model_type)
    print('size=',size)
    print('pac_num=',pac_num)

    return my_nums, level_line, slope, coef, mode_collapse,\
           model_structure, model_type, size, pac_num

files = list(range(341, 346))
my_nums, level_line, slope, coef, mode_collapse,model_structure, model_type, size, pac_num = extract_config(files)
accuracy_list = []
modes_unique_list = []
modes_generated_list = []

Time = []

accuracy_final_list = []
modes_unique_final_list = []
modes_generated_final_list = []

Incorrect_files = []
for index_files, num in enumerate(files):
    accuracy_temp = []
    modes_unique_temp = []
    modes_generated_temp = []
    print('file', num)
    my_lines = []
    # f = open(path + str(num) + '.txt', 'rt')

    f = open(path + str(num) + '/cout.txt', 'rt')
    for my_line in f:
        my_lines.append(my_line.rstrip('\n'))
    flag = 1
    for i in range(len(my_lines)):

        for num_epoch in range(1, 6):  # 40*1-1, ..., 40*9-1
        
            indexes = []
            if 'Epoch: [' + str(num_epoch * 40 - 1) + '/250]' in my_lines[i]:  # results per 40 epochs

                if mode_collapse[index_files] == "VarGAN":
                    # print('VARGAN')
                    if size[index_files] == 9:
                        for j in range(60):
                            # print(my_lines[i + j])
                            if ('accuracy_matrix out of 40000 quad' not in my_lines[i + j] and \
                                    'accuracy_matrix out of 40000' in my_lines[i + j]) :
                                # print(my_lines[i + j])
                                indexes.append(i+j)
                                indexes.append(i + j + 1)
                            if 'Configurations generated for class quad:' in my_lines[i + j]:
                                # print(my_lines[i + j])
                                indexes.append(i+j)

                    elif size[index_files] == 19:
                        for j in range(60):

                            if ('accuracy_matrix out of 40000 quad' in my_lines[i + j]):
                                # print(my_lines[i + j])
                                indexes.append(i + j)
                                indexes.append(i + j + 1)
                            if 'Configurations generated for class quad:' in my_lines[i + j]:
                                # print(my_lines[i + j])
                                indexes.append(i + j)

                else:
                    indexes = [i + 9, i + 10, i + 47, i + 49]
                for ir in indexes:
                    print(my_lines[ir])
                print('epochs',indexes)
                accuracy, num_modes_unique, num_modes_generated = extract_inline(my_lines, indexes)
                accuracy_temp.append(accuracy)
                modes_unique_temp.append(num_modes_unique)
                modes_generated_temp.append(num_modes_generated)

        if 'Completed after' in my_lines[i]:
            print(my_lines[i], get_digit(my_lines[i]))
            Time.append(get_digit(my_lines[i])[0] * 3600 + get_digit(my_lines[i])[1] * 60 + get_digit(my_lines[i])[2])
        if 'training finished' in my_lines[i]:  # only last results

            indexes = [i + 4, i + 5, i + 43, i + 45]
            for ir in indexes:
                print(my_lines[ir])
            accuracy, num_modes_unique, num_modes_generated = extract_inline(my_lines, indexes)
            accuracy_final_list.append(accuracy)
            modes_unique_final_list.append(num_modes_unique)
            modes_generated_final_list.append(num_modes_generated)
    print(Time)

    accuracy_list.append(accuracy_temp)
    modes_unique_list.append(modes_unique_temp)
    modes_generated_list.append(modes_generated_temp)

    f.close()

print('Incorrect files', Incorrect_files)
print('Time', Time)
print('final_accuracy=np.array(', accuracy_final_list, ')')
print('final_unique=np.array(', modes_unique_final_list, ')')
print('final_generated=np.array(', modes_generated_final_list, ')')

print('accuracy=np.array(', accuracy_list, ')')
print('unique=np.array(', modes_unique_list, ')')
print('generated=np.array(', modes_generated_list, ')')
