import torch
from main_sphere_ellipsoid import ex
import sacred
Num = 4
dataset_size = [100000]
size_wanted = [40000]
batch_size = [100]
num_epoch = 600
GAN_model = ['GAN']
for j in range(Num):
    for size_w in size_wanted:
        for data in dataset_size:
            for b in batch_size:
                for d, g in learning_rate:
                    torch.cuda.empty_cache()
                    ex.run(config_updates={'dataset_size':data, 'size_wanted':size_w,'batch_size':b,
                   'dimension_orig': [16,16,16], 'dimension': [16,16,16],
                   'num_epochs': num_epoch,'learning_rate_d':d, 'diversity':False,  'mode_collapse' : 'PacGAN',
                   'learning_rate_g':g,'GAN_model':GAN_model[0],'filled':True,'connectedFlag':False,
                   'itr_critic':5, 'dataset_include_tumor':False})



