# coding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import torchvision.utils as vutils


class Gen(nn.Module):
    def __init__(self, nbz, nbf, nc, nbgpu):
        super(Gen, self).__init__()
        self.nb_z = nbz
        self.nb_filters = nbf
        self.nb_channels = nc
        self.nb_gpu = nbgpu
        self.build_model()

    def build_model(self):
        # define a sequential net here
        self.net = nn.Sequential(
                        nn.ConvTranspose2d(self.nb_z, self.nb_filters * 8, 4, 1, 0, bias=False),
                        nn.BatchNorm2d(self.nb_filters * 8),
                        nn.ReLU(True),
                        # state size. (ngf*8) x 4 x 4
                        nn.ConvTranspose2d(self.nb_filters * 8, self.nb_filters * 4, 4, 2, 1, bias=False),
                        nn.BatchNorm2d(self.nb_filters * 4),
                        nn.ReLU(True),
                        # state size. (ngf*4) x 8 x 8
                        nn.ConvTranspose2d(self.nb_filters * 4, self.nb_filters * 2, 4, 2, 1, bias=False),
                        nn.BatchNorm2d(self.nb_filters * 2),
                        nn.ReLU(True),
                        # state size. (ngf*2) x 16 x 16
                        nn.ConvTranspose2d(self.nb_filters * 2, self.nb_filters, 4, 2, 1, bias=False),
                        nn.BatchNorm2d(self.nb_filters),
                        nn.ReLU(True),
                        # state size. (ngf) x 32 x 32
                        nn.ConvTranspose2d( self.nb_filters, self.nb_channels, 4, 2, 1, bias=False),
                        nn.Tanh())
    def forward(self,x):
        if isinstance(x.data, torch.cuda.FloatTensor) and self.nb_gpu > 1:
            output = nn.parallel.data_parallel(self.net, input, range(self.nb_gpu))
        else:
            output = self.net(x)
        return output



class Disc(nn.Module):
    def __init__(self,  nbf, nc, nbgpu):
        super(Disc, self).__init__()

        self.nb_filters = nbf
        self.nb_channels = nc
        self.nb_gpu = nbgpu
        # define a sequential net here
        self.build_model()

    def build_model(self):
        self.net = nn.Sequential(
                        # input is (nc) x 64 x 64
                        nn.Conv2d(self.nb_channels, self.nb_filters, 4, 2, 1, bias=False),
                        nn.LeakyReLU(0.2, inplace=True),
                        # state size. (ndf) x 32 x 32
                        nn.Conv2d(self.nb_filters, self.nb_filters * 2, 4, 2, 1, bias=False),
                        nn.BatchNorm2d(self.nb_filters * 2),
                        nn.LeakyReLU(0.2, inplace=True),
                        # state size. (ndf*2) x 16 x 16
                        nn.Conv2d(self.nb_filters * 2, self.nb_filters * 4, 4, 2, 1, bias=False),
                        nn.BatchNorm2d(self.nb_filters * 4),
                        nn.LeakyReLU(0.2, inplace=True),
                        # state size. (ndf*4) x 8 x 8
                        nn.Conv2d(self.nb_filters * 4, self.nb_filters * 8, 4, 2, 1, bias=False),
                        nn.BatchNorm2d(self.nb_filters * 8),
                        nn.LeakyReLU(0.2, inplace=True),
                        # state size. (ndf*8) x 4 x 4
                        nn.Conv2d(self.nb_filters * 8, 1, 4, 1, 0, bias=False),
                        nn.Sigmoid())
    def forward(self,x):
        if isinstance(x.data, torch.cuda.FloatTensor) and self.nb_gpu > 1:
            output = nn.parallel.data_parallel(self.net, input, range(self.nb_gpu))
        else:
            output = self.net(x)
        return output.view(-1, 1)
