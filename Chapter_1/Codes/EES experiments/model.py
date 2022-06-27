import torch
from torch import nn, optim
import numpy as np
from torch.autograd.variable import Variable

def initialize_models(Model, Model_structure, Model_type, full_image, Full_len, Losses, args):
    
    if Model == 'Normal':
        if Model_structure == 'FF':
            if Model_type == 'Conditional':
                from Models.Conditional_vanilla_GAN import DiscriminatorNet, GeneratorNet
            else:
                from Models.Vanilla_GAN import DiscriminatorNet, GeneratorNet

        else:
            if full_image:
                from Model_19_Full import DiscriminatorNet, GeneratorNet
            else:
                if Full_len == 19:
                    from Model_19_partial_Type2 import DiscriminatorNet, GeneratorNet
                elif Full_len == 9:
                    from models_type2 import DiscriminatorNet, GeneratorNet
    elif Model == 'Complex':
        from Model_9_Partial_Type2 import DiscriminatorNet, GeneratorNet


    [channels, Channel_factor, Kernel_factor, zdim, num_features, num_condition, num_pixels] = args

    if Model_structure == 'FF':
        if Model_type == 'Conditional':
            discriminator = DiscriminatorNet(num_features)
        else:
            discriminator = DiscriminatorNet(num_pixels)

        generator = GeneratorNet(num_condition, num_pixels)
    else:
        discriminator = DiscriminatorNet(args)
        generator = GeneratorNet(args)

    if torch.cuda.is_available():
        discriminator.cuda()
        generator.cuda()

    # Optimizers
    d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002)
    g_optimizer = optim.Adam(generator.parameters(), lr=0.0002)

    # Loss functions
    loss_dict = {}

    for i in Losses:
        if i == 'BCE':
            #  BCE Loss(Creates a criterion that measures the Binary Cross Entropy between the target and the output)
            loss_dict['BCE'] = nn.BCELoss()
        if i == 'symmetric':
            # Symmetric Loss
            loss_dict['SYM'] =  LossSymmetry()

    return generator, discriminator, g_optimizer, d_optimizer, loss_dict
