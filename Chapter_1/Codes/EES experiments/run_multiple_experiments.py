from Main import ex
import sacred
# from Veegan_Main import ex

# from Classifier import ClassifierNet,ClassifierNet_19_partial

# config = {'num_epochs':2000, 'categorization':2, 'full_image':True}
# config = {'num_epochs':2000, 'categorization':2, 'full_image':True, 'partial_2fold':False}
# ex.run()
Num = 5
# for i in range(Num):
#     ex.run(config_updates={'num_epochs':400, 'Model_type':'Vanilla'})
# for i in range(Num):
#     ex.run(config_updates={'num_epochs': 400,'Model_type':'Conditional' ,'mode_collapse':''})
# for i in range(Num):
#     ex.run(config_updates={'num_epochs': 400,'Model_type':'Conditional' ,'mode_collapse':'Minibatch'})
# for i in range(Num):
#     ex.run(config_updates={'num_epochs': 400, 'Model_type': 'Conditional', 'mode_collapse': 'DSGAN'})
# for i in range(Num):
#     ex.run(config_updates={'num_epochs': 400, 'Model_type': 'Conditional', 'mode_collapse': 'PacGAN'})
#
Num = 5
num_epochs = 400
# for i in range(Num):
#     print('GAN', i, 19)
#     ex.run(config_updates={'Model_type':'Vanilla','num_epochs':400,
#                            'mode_collapse':'', 'categorization':2.3,
#
#                            'Pixel_Full':19})
  # ['FF','Convolutional', 'ConvOriginal']

# for i in range(Num):
#     print('PacGAN 19', i)
#     ex.run(config_updates={'Model_structure':'Convolutional', 'Model_type':'Conditional',
#                            'num_epochs':400, 'mode_collapse':'PacGAN',
#                            'categorization':2.3,
#
#                            'Pixel_Full':19})

# for i in range(Num):
#     print(' 9 8 cat', i)
#     ex.run(config_updates={'Model_structure':'Convolutional', 'Model_type':'Conditional',
#                            'num_epochs':400, 'mode_collapse':'',
#                            'categorization':8,
#                            'Pixel_Full':9})

for i in range(Num):
    print(' 9 8 cat pacGAN compare', i)
    ex.run(config_updates={'Model_structure':'Convolutional', 'Model_type':'Conditional',
                           'num_epochs':400, 'mode_collapse':'PacGAN','compare':True,
                           'categorization':8,
                           'Pixel_Full':9})
for i in range(Num):
    print(' 9 8 cat', i)
    ex.run(config_updates={'Model_structure':'Convolutional', 'Model_type':'Conditional',
                           'num_epochs':400, 'mode_collapse':'','compare':True,
                           'categorization':8,
                           'Pixel_Full':9})

# for i in range(Num):
#     print('DSGAN 19', i)
#     ex.run(config_updates={'Model_structure':'Convolutional', 'Model_type':'Conditional',
#                            'num_epochs':400, 'mode_collapse':'DSGAN',
#                            'categorization':2.3,
#                            'Pixel_Full':19})
# for i in range(Num):
#     print('DSGAN 9', i)
#     ex.run(config_updates={'Model_structure':'Convolutional', 'Model_type':'Conditional',
#                            'num_epochs':400, 'mode_collapse':'DSGAN',
#                            'categorization':2,
#                            'Pixel_Full':9})
# # for i in range(Num):
#     print('Minibatch', i)
#     ex.run(config_updates={'Model_type':'Vanilla','num_epochs': 400, 'categorization':2.3,
#                            'Pixel_Full':19,'mode_collapse': 'Minibatch'})
# for i in range(Num):
#     print('DSGAN', i)
#     ex.run(config_updates={'Model_type':'Vanilla','num_epochs': 400, 'mode_collapse':'DSGAN', 'categorization':2.3,
#                            'Pixel_Full':19})



# for i in range(Num):
#     ex.run(config_updates={'num_epochs': num_epochs, 'mode_collapse': '', 'categorization': 2.3, 'Pixel_Full': 19})
# for i in range(Num):
#     ex.run(config_updates={'num_epochs': num_epochs, 'mode_collapse': 'Minibatch', 'categorization': 2.3, 'Pixel_Full': 19})

#

# for i in range(Num):
#     ex.run(config_updates={'num_epochs': 400,'Model_type':'Vanilla' ,'mode_collapse':''})
# for i in range(Num):
#     ex.run(config_updates={'num_epochs': 400,'Model_type':'Vanilla' ,'mode_collapse':'Minibatch'})
# for i in range(Num):
#     ex.run(config_updates={'num_epochs': 400, 'Model_type': 'Vanilla', 'mode_collapse': 'DSGAN'})
# for i in range(Num):
#     ex.run(config_updates={'num_epochs': 400, 'Model_type': 'Vanilla', 'mode_collapse': 'PacGAN'})
#
# for i in range(Num):
#     ex.run(config_updates={'num_epochs': 400,'Model_type':'Conditional' ,'mode_collapse':'','categorization':8})
# for i in range(Num):
#     ex.run(config_updates={'num_epochs': 400,'Model_type':'Conditional' ,'mode_collapse':'Minibatch','categorization':8})
# for i in range(Num):
#     ex.run(config_updates={'num_epochs': 400, 'Model_type': 'Conditional', 'mode_collapse': 'DSGAN','categorization':8})
# for i in range(Num):
#     ex.run(config_updates={'num_epochs': 400, 'Model_type': 'Conditional', 'mode_collapse': 'PacGAN','categorization':8})
#
# for i in range(Num):
#     ex.run(config_updates={'num_epochs': 400,'Model_type':'Vanilla' ,'mode_collapse':'','categorization':8})
# for i in range(Num):
#     ex.run(config_updates={'num_epochs': 400,'Model_type':'Vanilla' ,'mode_collapse':'Minibatch','categorization':8})
# for i in range(Num):
#     ex.run(config_updates={'num_epochs': 400, 'Model_type': 'Vanilla', 'mode_collapse': 'DSGAN','categorization':8})
# for i in range(Num):
#     ex.run(config_updates={'num_epochs': 400, 'Model_type': 'Vanilla', 'mode_collapse': 'PacGAN','categorization':8})

# for i in range(2):
#
#     for j in range(Num):
#         print('Vanilla')
#         ex.run(config_updates={'Model_type': 'Vanilla'})
#
#     for j in range(Num):
#         print('Conditional')
#         ex.run(config_updates={'Model_type':'Conditional'})

    # if i == 1:
        # for j in range(Num):
            # ex.run(config_updates={'Model': 'Normal', 'n_critic': 1,'categorization':2.3, 'Pixel_Full':19})
#
