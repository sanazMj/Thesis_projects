import sacred
import torch

# from GAN_synthetic_dataata import ex


Num = 5
num_epoch = 50

for j in range(Num):
     print('GAN ring', j)
     torch.cuda.empty_cache()
     ex.run(config_updates={'num_epochs': num_epoch, 'pac_dim':1, 'dataset': 'ring', 'n_mixture':8})
