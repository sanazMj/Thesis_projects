

import torch

Num = 5
num_epoch = 30
#
from GAN import ex
# for j in range( Num):
#     print('GAN',j, num_epoch)
#
#     torch.cuda.empty_cache()
#     ex.run(config_updates={ 'num_epochs': num_epoch,  'pac_dim':1})
# for j in range( Num):
#     print('PacGAN',j, num_epoch)
#
#     torch.cuda.empty_cache()
#     ex.run(config_updates={ 'num_epochs': num_epoch,  'pac_dim':4})
#


from VARNET import ex
# # from VARNET_with_labels import ex
# for j in range(Num):
#     print('varnet',j,30 )
#
#     torch.cuda.empty_cache()
#     ex.run(config_updates={ 'num_epochs': 30,  'pac_dim':1, 'Model_structure': 'FF'})

for j in range(1, Num):
    print('Pacvarnet', j, 30)

    torch.cuda.empty_cache()
    ex.run(config_updates={'num_epochs': 30, 'pac_dim': 4, 'Model_structure': 'FF'})

for j in range(Num):
    print('varnet',j,30,'Convolutional' )

    torch.cuda.empty_cache()
    ex.run(config_updates={ 'num_epochs': 30,  'pac_dim':1, 'Model_structure': 'Convolutional'})

for j in range(Num):
    print('Pacvarnet', j, 30, 'Convolutional')

    torch.cuda.empty_cache()
    ex.run(config_updates={'num_epochs': 30, 'pac_dim': 4, 'Model_structure': 'Convolutional'})
#
# #
#
# from VARNET import ex
# for j in range(Num):
#     print('pacvarnet',j, 20)
#
#     torch.cuda.empty_cache()
#     ex.run(config_updates={ 'num_epochs': 20,  'pac_dim':4})