
# coding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import torchvision.utils as vutils


BATCH_SIZE = 32
nb_hidden = 10
nb_epochs = 10
lr = 0.001
momentum = 0.5
cuda = True
log_interval = 10
dataset_name = 'cifar10'
dataroot = './data'
image_h = 64
image_w = 64
kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
out_dir = './samples/'




if dataset_name in ['imagenet', 'folder', 'lfw']:
    # folder dataset
    dataset = datasets.ImageFolder(root=dataroot,
                               transform=transforms.Compose([
                                   transforms.Scale(image_h),
                                   transforms.CenterCrop(image_w),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
elif dataset_name == 'lsun':
    dataset = datasets.LSUN(db_path=dataroot, classes=['bedroom_train'],
                        transform=transforms.Compose([
                            transforms.Scale(image_h),
                            transforms.CenterCrop(image_w),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                        ]))
elif dataset_name == 'cifar10':
    dataset = datasets.CIFAR10(root=dataroot, download=True,
                           transform=transforms.Compose([
                               transforms.Scale(image_h),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ])
    )
elif dataset_name == 'mnist':
    dataset = datasets.MNIST(root=dataroot, train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])
    )
assert dataset
dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE,
                                         shuffle=True, **kwargs)

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


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



gen = Gen(nb_hidden, 32, 3, 1)
gen.apply(weights_init)



disc = Disc(32, 3, 1)
disc.apply(weights_init)



# loss 
criterion = nn.BCELoss()


# inputs 
input = torch.FloatTensor(BATCH_SIZE, 3, image_h, image_w)
# noise 
noise = torch.FloatTensor(BATCH_SIZE, nb_hidden, 1, 1)
fixed_noise = torch.FloatTensor(BATCH_SIZE, nb_hidden, 1, 1).normal_(0, 1)

# labels
label = torch.FloatTensor(BATCH_SIZE)
real_label = 1
fake_label = 0



if cuda:
    gen.cuda()
    disc.cuda()
    criterion.cuda()
    input, label = input.cuda(), label.cuda()
    noise, fixed_noise = noise.cuda(), fixed_noise.cuda()



input = Variable(input)
label = Variable(label)
noise = Variable(noise)
fixed_noise = Variable(fixed_noise)



# setup optimizer
optimizerD = optim.Adam(disc.parameters(), lr=lr, betas=(0.5, 0.999))
optimizerG = optim.Adam(gen.parameters(), lr=lr, betas=(0.5, 0.999))



for epoch in range(10):
    for i, data in enumerate(dataloader, 0):
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        # train with real
        disc.zero_grad()
        real_cpu, _ = data
        batch_size = real_cpu.size(0)
        input.data.resize_(real_cpu.size()).copy_(real_cpu)
        label.data.resize_(batch_size).fill_(real_label)

        output = disc(input)
        errD_real = criterion(output, label)
        errD_real.backward()
        D_x = output.data.mean()

        # train with fake
        noise.data.resize_(batch_size, nb_hidden, 1, 1)
        noise.data.normal_(0, 1)
        fake = gen(noise)
        label.data.fill_(fake_label)
        output = disc(fake.detach())
        errD_fake = criterion(output, label)
        errD_fake.backward()
        D_G_z1 = output.data.mean()
        errD = errD_real + errD_fake
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        gen.zero_grad()
        label.data.fill_(real_label)  # fake labels are real for generator cost
        output = disc(fake)
        errG = criterion(output, label)
        errG.backward()
        D_G_z2 = output.data.mean()
        optimizerG.step()

        print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
              % (epoch, nb_epochs, i, len(dataloader),
                 errD.data[0], errG.data[0], D_x, D_G_z1, D_G_z2))
        if i % 100 == 0:
            vutils.save_image(real_cpu,
                    '%s/real_samples.png' % out_dir,
                    normalize=True)
            fake = gen(fixed_noise)
            vutils.save_image(fake.data,
                    '%s/fake_samples_epoch_%03d.png' % (out_dir, epoch),
                    normalize=True)

    # do checkpointing
    torch.save(gen.state_dict(), '%s/netG_epoch_%d.pth' % (out_dir, epoch))
    torch.save(disc.state_dict(), '%s/netD_epoch_%d.pth' % (out_dir, epoch))



