"""
Neural network definition for both Generator and discriminator
"""
import torch 
import torch.nn as nn 
import torch.nn.functional as F 


def simple_deconv(in_planes, out_planes, k_size=3, stride=1, padding=0):
    return nn.Sequential(
        nn.ConvTranspose2d(in_planes, out_planes,
                kernel_size=k_size, stride=stride, 
                padding=padding, bias=False),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(inplace=True)
    )

def simple_conv(in_planes, out_planes, k_size=3, stride=1, padding=0):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes,kernel_size=k_size,
                stride=stride, padding=padding, bias=False),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(inplace=True)
    )

class Generator(nn.Module):
    r""" Generator network class 
    params:
    """
    def __init__(self, nz, ngf, nc):
        super(Generator,self).__init__()
        layers = []
        layers.append(simple_deconv(nz, ngf*8, 4, 1, 0))
        layers.append(simple_deconv(ngf*8, ngf*4, 4, 2, 1))
        layers.append(simple_deconv(ngf*4, ngf*2, 4, 2, 1))
        layers.append(simple_deconv(ngf*2, ngf, 4, 2, 1))
        layers.append(nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False))
        layers.append(nn.Tanh())
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        out = self.main(x)
        return out

class Disciminator(nn.Module):
    r"""Discriminator network class"""
    def __init__(self,nc, ndf,N=1):
        super(Disciminator,self).__init__()
        layers = []
        layers.append(nn.Conv2d(nc, ndf, 4, 2, 1, bias=False))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        layers.append(simple_conv(ndf, ndf*2,4,2,1))
        layers.append(simple_conv(ndf*2, ndf*4, 4, 2, 1))
        layers.append(simple_conv(ndf*4, ndf*8, 4, 2, 1))
        layers.append(nn.Conv2d(ndf*8,N,4,1,0,bias=False))
        layers.append(nn.Sigmoid())
        self.main = nn.Sequential(*layers)
    
    def forward(self,x):
        out = self.main(x)
        return out.view(-1, 1).squeeze(1)
