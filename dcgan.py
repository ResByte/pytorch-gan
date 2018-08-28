# coding: utf-8
from __future__ import print_function
import argparse
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import visdom
from networks import _netG, _netD
from utils import *

# setup config for parameters
opt = config()

# create dataset loader 
dataset = get_dataset(opt)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                         shuffle=True, num_workers=int(opt.workers))

ngpu = int(opt.ngpu)      # number of gpus 
nz = int(opt.nz)          # z dimensions
ngf = int(opt.ngf)        # generator filter size
ndf = int(opt.ndf)        # discrimintator filter size 
nc = 3                    # dataset image channels 

# setup Generator 
netG = _netG(ngpu, nz, ngf, nc)
netG.apply(weights_init)
if opt.netG != '':
    netG.load_state_dict(torch.load(opt.netG))
print(netG)

# setup discriminator 
netD = _netD(ngpu, ndf, nc)
netD.apply(weights_init)
if opt.netD != '':
    netD.load_state_dict(torch.load(opt.netD))
print(netD)

# TODO: update according to v0.4
if opt.cuda:
    netD = netD.cuda()
    netG = netG.cuda()
    
# loss used is Binary x-entropy
criterion = nn.BCELoss()

# variables 
fixed_noise = torch.FloatTensor(opt.batchSize, nz, 1, 1).normal_(0, 1)

# TODO: update according to v0.4 
if opt.cuda:
    fixed_noise = fixed_noise.cuda()

fixed_noise = Variable(fixed_noise)

# setup optimizer, Adam for faster convergence
optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

# visdom is used for visualization of results
vis = visdom.Visdom()
lot = vis.line(
            X=torch.zeros((1,)).cpu(),
            Y=torch.zeros((1, 2)).cpu(),
            opts=dict(
                xlabel='Iterations',
                ylabel='Loss',
                title='Current Losses',
                legend=['Gen Loss', 'Disc Loss']
        ) )

# training loop
count = 0
for epoch in range(opt.niter):
    for i, (inputs, targets) in enumerate(dataloader, 0):
        count +=1
        if opt.cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs), Variable(targets)
        batch_size = inputs.size(0)
        
        zeros = Variable(torch.zeros(batch_size))
        ones = Variable(torch.ones(batch_size))
        
        if opt.cuda:
            zeros, ones = zeros.cuda(), ones.cuda()
        
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        optimizerD.zero_grad()
        
        # train with real :  log(D(x))
        d_real = netD(inputs)               # forward pass
        errD_real = criterion(d_real, ones) # compare against labels for real
        errD_real.backward()                # backward pass
        D_x = d_real.data.mean()            # statistics
        

        # train with fake : log(1 - D(G(z)))
        # create minibatch noise
        minibatch_noise = Variable(
                            torch.from_numpy(
                                np.random.randn(batch_size,nz, 1, 1).astype(np.float32)))
        if opt.cuda:
            minibatch_noise = minibatch_noise.cuda()
        
        g_fake = netG(minibatch_noise).detach()    # create image from random noise
        d_fake = netD(g_fake)                      # use detached image as input
        errD_fake = criterion(d_fake, zeros)       # we want these to be as fake 
        errD_fake.backward()                       # propagate loss
        D_G_z1 = d_fake.data.mean()                # statistics
        
        errD = errD_real + errD_fake
        optimizerD.step()
        
        # if there is need for clamping 
        # for p in netD.parameters():
        #     p.data.clamp_(-0.01, 0.01)

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################

        optimizerG.zero_grad()
        
        # sample new noise
        # create minibatch noise
        minibatch_noise = Variable(
                            torch.from_numpy(
                                np.random.randn(batch_size,nz, 1, 1).astype(np.float32)))
        if opt.cuda:
            minibatch_noise = minibatch_noise.cuda()
        
        g_fake = netG(minibatch_noise)          # get fake image from G 
        output = netD(g_fake)                   # classify it using D 
        errG = criterion(output, ones)          # get loss as if it is real
        errG.backward()                         # propagate error
        D_G_z2 = output.data.mean()
        optimizerG.step()

        print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
              % (epoch, opt.niter, i, len(dataloader),
                 errD.data[0], errG.data[0], D_x, D_G_z1, D_G_z2))
        vis.line(
                    X=torch.ones((1, 2)).cpu()*count,
                    Y=torch.Tensor([errG.data[0],errD.data[0]]).unsqueeze(0).cpu(),
                    win=lot,
                    update='append'
                )
        if i % 100 == 0:
            # vutils.save_image(real_cpu,
            #         '%s/real_samples.png' % opt.outf,
            #         normalize=True)
            fake = netG(fixed_noise)
            
            grid = vutils.make_grid(fake.cpu().data)
            ndarr = grid.mul(255).clamp(0, 255).byte().numpy()

            vis.image(ndarr)
            # vutils.save_image(fake.data,
            #         '%s/fake_samples_epoch_%03d.png' % (opt.outf, epoch),
            #         normalize=True)

    # if do checkpointing
    # torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (opt.outf, epoch))
    # torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (opt.outf, epoch))
