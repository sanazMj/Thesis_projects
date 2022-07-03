import torch
import torch.autograd as autograd
from torch.autograd.variable import Variable
import numpy as np
from GAN_3D.helper import sofSparsity, shannon_diversity, connected, connected_target_1
LAMBDA = 10


def cal_acc(zeros, ones):
    accuracy = 0.0
    zeros = zeros.detach().cpu().numpy()
    ones = ones.detach().cpu().numpy()
    for example in zeros:
        if not np.isnan(example[0]):
            if example[0] < 0.5:
                accuracy += 1.0

    for example in ones:
        if not np.isnan(example[0]):
            if example[0] > 0.5:
                accuracy += 1.0
    accuracy = accuracy / (float(len(zeros) + len(ones)))
    # print('The accuracy of the discriminator is: ' + str(accuracy))
    return accuracy
def calc_gradient_penalty(netD, real_data, fake_data, use_cuda=True):
    #print real_data.size()
    x = torch.ones(real_data.shape)
    alpha = torch.rand(real_data.shape[0], 1)
    for i in range(real_data.shape[0]):
        x[i,:,:,:,:] =  x[i,:,:,:,:]* alpha[i,:]
    alpha = x
    alpha = alpha.cuda() if use_cuda else alpha

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    if use_cuda:
        interpolates = interpolates.cuda()
    interpolates = autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda() if use_cuda else torch.ones(
                                  disc_interpolates.size()),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty

def ones_target(size):
    '''
    Tensor containing ones, with shape = size
    '''
    data = Variable(torch.ones(size, 1))
    return data


def zeros_target(size):
    '''
    FAKE data
    Tensor containing zeros, with shape = size
    '''
    data = Variable(torch.zeros(size, 1))
    return data

def train_discriminator_RaSGAN(discriminator, optimizer, real_data, fake_data, valid, fake, loss):
    # Discriminator output is not sigmoid
    optimizer.zero_grad()

    real_pred = discriminator(real_data)
    fake_pred = discriminator(fake_data)

    d_loss = (loss(real_pred - torch.mean(fake_pred), valid) +
            loss(fake_pred - torch.mean(real_pred), fake)) / 2

    d_loss.backward()
    optimizer.step()
    return d_loss, [], [],[],[]

def train_generator_RaSGAN(discriminator, optimizer, real_data, fake_data, valid,fake, loss):
    # Discriminator output is not sigmoid
    optimizer.zero_grad()
    real_pred = discriminator(real_data)
    fake_pred = discriminator(fake_data)
    g_loss = (loss(real_pred - torch.mean(fake_pred), fake) +
            loss(fake_pred - torch.mean(real_pred), valid)) / 2

    g_loss.backward()
    optimizer.step()
    return g_loss, [], []

def train_discriminator_WGAN_GP(discriminator, optimizer, real_data, fake_data, valid, fake, loss, d_thresh, soft=False):
    # Discriminator output is not sigmoid

    sparse_coef = 10
    optimizer.zero_grad()

    real_pred = discriminator(real_data)
    fake_pred = discriminator(fake_data)
    d_real_loss =  torch.mean(real_pred)
    d_fake_loss =  torch.mean(fake_pred)


    gradient_penalty = calc_gradient_penalty(discriminator, real_data, fake_data)
    d_loss = d_fake_loss - d_real_loss + gradient_penalty
    d_sparse1 = []
    if soft:

        d_sparse = sofSparsity(fake_data.detach().cpu().numpy())
        b = torch.tensor(np.array(d_sparse))
        d_sparse = autograd.Variable(b.cuda(), requires_grad=True)
        d_sparse1 = torch.mean(d_sparse)
        d_loss += d_sparse1 * sparse_coef
        # print('d_sparse', torch.mean(d_sparse))


    d_loss.backward()
    optimizer.step()

    return d_loss, d_real_loss, d_fake_loss,d_real_loss, d_fake_loss,  [], d_sparse1

def train_generator_WGAN_GP(discriminator, optimizer, real_data, fake_data, valid,fake, loss, diversity=False, target_points=7,connectedFlag=False):
    # Discriminator output is not sigmoid
    div_coef = 1
    d_div1 = []
    connect_coef = 1

    optimizer.zero_grad()
    fake_pred = discriminator(fake_data)
    g_loss =  -torch.mean(fake_pred)
    d_connect = []
    KLD = 0.0
    if connectedFlag:
        d_connect = connected_target_1(fake_data.detach().cpu())
        # b = torch.tensor(np.array(results))
        # d_connect = autograd.Variable(b.cuda(), requires_grad=True)
        # d_connect = torch.mean(d_connect)
        # KLD = torch.tensor(np.array(KLD))
        # KLD = autograd.Variable(KLD.cuda(), requires_grad=True)
        # KLD = torch.mean(KLD)
        g_loss += (d_connect) * connect_coef
    else:
        d_connect = 0.0
        KLD = 0.0
    if diversity:

        d_div = shannon_diversity(fake_data.detach().cpu().numpy(), target_points)
        b = torch.tensor(np.array(d_div))
        d_div = autograd.Variable(b.cuda(), requires_grad=True)
        d_div1 = torch.mean(d_div)
        g_loss += (1-d_div1) * div_coef
        # print('d_div', torch.mean(d_div))
    g_loss.backward()
    optimizer.step()
    return g_loss, [], g_loss, d_connect, KLD, d_div1

def train_discriminator_LSGAN(discriminator, optimizer, real_data, fake_data, valid, fake, loss):
    # Discriminator output is not sigmoid
    optimizer.zero_grad()

    real_pred = discriminator(real_data)
    fake_pred = discriminator(fake_data)
    d_real_loss =  torch.mean((real_pred - valid) ** 2)
    d_fake_loss =  torch.mean((fake_pred) ** 2)
    d_loss = d_real_loss + d_fake_loss
    d_loss.backward()
    optimizer.step()
    return d_loss, d_real_loss, d_fake_loss,d_real_loss, d_fake_loss

def train_generator_LSGAN(discriminator, optimizer, real_data, fake_data, valid,fake, loss):
    # Discriminator output is not sigmoid
    optimizer.zero_grad()
    fake_pred = discriminator(fake_data)
    g_loss =  torch.mean((fake_pred-valid) ** 2)

    g_loss.backward()
    optimizer.step()
    return g_loss, [], []

def train_discriminator_RSGAN(discriminator, optimizer, real_data, fake_data, valid, fake, loss):
    # Discriminator output is not sigmoid
    optimizer.zero_grad()

    real_pred = discriminator(real_data)
    fake_pred = discriminator(fake_data)
    d_loss = loss(real_pred - fake_pred, valid)

    d_loss.backward()
    optimizer.step()
    return d_loss, [], [],[],[]

def train_generator_RSGAN(discriminator, optimizer, real_data, fake_data, valid,fake, loss):
    # Discriminator output is not sigmoid
    optimizer.zero_grad()
    real_pred = discriminator(real_data)
    fake_pred = discriminator(fake_data)
    g_loss = loss(fake_pred - real_pred , valid)

    g_loss.backward()
    optimizer.step()
    return g_loss, [], []

def train_discriminator_GAN(discriminator, optimizer, real_data, fake_data, valid, fake, loss,d_thresh, soft=False):
    sparse_coef = 10
    # Measure discriminator's ability to classify real from generated samples
    real_prediction = discriminator(real_data)
    fake_prediction = discriminator(fake_data)

    real_loss = loss(real_prediction, valid)
    fake_loss = loss(fake_prediction, fake)
    d_loss = (real_loss + fake_loss) / 2

    d_real_acu = torch.ge(real_prediction.squeeze(), 0.5).float()
    d_fake_acu = torch.le(fake_prediction.squeeze(), 0.5).float()
    d_total_acu = torch.mean(torch.cat((d_real_acu, d_fake_acu), 0))
    d_sparse1 = []
    if soft:
        d_sparse = sofSparsity(fake_data.detach().cpu().numpy())
        b = torch.tensor(np.array(d_sparse))
        d_sparse = autograd.Variable(b.cuda(), requires_grad=True)
        d_sparse1 = torch.mean(d_sparse)
        d_loss += d_sparse1 * sparse_coef

    if d_total_acu <= d_thresh:

        optimizer.zero_grad()
        d_loss.backward()
        optimizer.step()
    return d_loss, real_prediction, fake_prediction,real_loss, fake_loss, d_total_acu, d_sparse1

def train_generator_GAN(discriminator, optimizer, real_imgs, fake_data, valid, fake, loss,diversity=False,target_points=7,  connectedFlag=False, connect_coef = 0.01):
    div_coef = 1
    d_div1 = []
    d_connect =[]
    optimizer.zero_grad()
    fake_pred = discriminator(fake_data)
    g_loss = loss(fake_pred, valid)
    if connectedFlag:
        results, KLD = connected(fake_data.detach().cpu())
        b = torch.tensor(np.array(results))
        d_connect = autograd.Variable(b.cuda(), requires_grad=True)
        d_connect = torch.mean(d_connect)
        KLD = torch.tensor(np.array(KLD))
        KLD = autograd.Variable(KLD.cuda(), requires_grad=True)
        KLD = torch.mean(KLD)
        g_loss += (d_connect  + KLD)* connect_coef
    else:
        d_connect = 0.0
        KLD = 0.0
    if diversity:
        d_div = shannon_diversity(fake_data.detach().cpu().numpy(), target_points)
        b = torch.tensor(np.array(d_div))
        d_div = autograd.Variable(b.cuda(), requires_grad=True)
        d_div1 = torch.mean(d_div)
        g_loss += (1 - d_div1) * div_coef
    g_loss.backward()
    optimizer.step()
    return g_loss, [], fake_pred,  d_connect , KLD, d_div1


'''

def train_discriminator(discriminator, optimizer, real_data, fake_data, loss):
    cuda = next(discriminator.parameters()).is_cuda
    N = real_data.size(0)
    # Reset gradients
    optimizer.zero_grad()
    # 1.1 Train on Real Data
    prediction_real = discriminator(real_data)
    # Calculate error and backpropagate
    target_real = ones_target(N)
    if cuda:
        target_real.cuda()

    error_real = loss(prediction_real, target_real)
    error_real.backward()

    # 1.2 Train on Fake Data
    prediction_fake = discriminator(fake_data)
    # Calculate error and backpropagate
    target_fake = zeros_target(N)
    if cuda:
        target_fake.cuda()
    error_fake = loss(prediction_fake, target_fake)
    error_fake.backward()

    # 1.3 Update weights with gradients
    optimizer.step()

    # Return error and predictions for real and fake inputs
    return error_real + error_fake, prediction_real, prediction_fake


def train_generator(discriminator, optimizer, fake_data, loss):
    cuda = next(discriminator.parameters()).is_cuda
    N = fake_data.size(0)  # Reset gradients
    optimizer.zero_grad()  # Sample noise and generate fake data
    prediction = discriminator(fake_data)  # Calculate error and backpropagate
    target = ones_target(N)
    if cuda:
        target.cuda()

    error = loss(prediction, target)
    error.backward()  # Update weights with gradients
    optimizer.step()  # Return error
    return error
'''