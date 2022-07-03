import torch
from Main_AlphaWGAN_connectedspheres import ex
import sacred
Num = 2
for i in range(Num):
     ex.run(config_updates={'cuda_device':1, 'dataset_kind':'Matlab', 'connected_loss_added':True})



