import numpy as np
import torch
import torchvision
import time
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision.utils as vutils
import argparse 
import visdom



def parse_args():
    parser = argparse.ArgumentParser(description='Wasserstein GAN')
    parser.add_argument('--dataset', type=str, default='FashionMNIST', help='the dataset')
    parser.add_argument('--dataroot', type=str, default='', help='path to dataset root')
    parser.add_argument('--epochs', type=int, default=10, help='nb epochs')
    parser.add_argument('--lr', type=float, default=3.0e-4, help='learning rate')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam')        
    parser.add_argument('--batch-size', type=int, default=32, help='batch size default 32')
    parser.add_argument('--save-dir', type=str, default='saves', help='output model directory')
    parser.add_argument('--result-dir', type=str, default='results', help='output sample images')
    parser.add_argument('--log-dir', type=str, default='logs', help='training log directory')
    parser.add_argument('--cuda', action='store_true', default=False, help='enable CUDA ')
    parser.add_argument('--seed', type=int, default=0, metavar='S', help='random seed')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N', help='logging step')
    parser.add_argument('--nz', type=int, default=128, help='hidden variable z dim')
    parser.add_argument('--ngf', type=int, default=64, help='generator filters size')
    parser.add_argument('--ndf', type=int, default=64, help='discriminator filter size')
    parser.add_argument('--ngpu', type=int, default=1, help='discriminator filter size')
    parser.add_argument('--noise-std', type=float, default=0.01, help='noise standard deviation')

    args = parser.parse_args()
    # args.cuda = not args.cuda and torch.cuda.is_available()
    
    if args.dataroot == '':
        args.dataroot = './data/{}'.format(args.dataset)

    args.modelname = 'WGAN'

    return args

# as the name says
def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)



# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

# as the name says
def initialize_weights(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.ConvTranspose2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
class View(nn.Module):
    def __init__(self, *sizes):
        super(View, self).__init__()
        self.sizes = sizes
    def forward(self, x):
        return x.view(*self.sizes)
    def __repr__(self):
        s = ('{name}({input_shape})')
        return s.format(name=self.__class__.__name__, input_shape=tuple(self.sizes))

# taken and modified from pytorch examples
# (https://github.com/pytorch/examples/blob/master/dcgan/main.py)
class _G(nn.Module):
    """
    Generater module 
    """
    def __init__(self, ngpu, nz, ngf, nc, nsize):
        super(_G, self).__init__()
        self.ngpu = ngpu # nb of gpus 
        self.nz = nz # size of latent dim
        self.nsize = nsize # size of image
        self.ngf = ngf # filter size
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            # nn.ConvTranspose2d(     nz, ngf * 8, 4, 1, 0, bias=False),
            # nn.BatchNorm2d(ngf * 8),
            # nn.ReLU(True),
            # # state size. (ngf*8) x 4 x 4
            # nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(ngf * 4),
            # nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.LeakyReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2,     ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(    ngf,      nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )
        self.fc = nn.Sequential(
            nn.Linear(nz, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, self.ngf*4 * (nsize // 8) * (nsize // 8)),
            nn.BatchNorm1d(self.ngf*4 * (nsize // 8) * (nsize // 8)),
            nn.ReLU(),
        )

    def forward(self, input):
        
        output = self.fc(input)
        output = self.main(output.view((-1,self.ngf*4, self.nsize//8, self.nsize//8)))
        return output

# Taken and modified from pytorch examples
# https://github.com/pytorch/examples/blob/master/dcgan/main.py
class _D(nn.Module):
    def __init__(self, ngpu, ndf, nc, nout, nsize):
        super(_D, self).__init__()
        self.ngpu = ngpu
        self.nsize = nsize
        self.ndf = ndf
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            # nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(ndf * 8),
            # nn.LeakyReLU(0.2, inplace=True),
            # # state size. (ndf*8) x 4 x 4
            # nn.Conv2d(ndf * 8, nout, 4, 1, 0, bias=False),
            # nn.Sigmoid()
        )
        self.fc = nn.Sequential(
            nn.Linear(self.ndf*4* ( nsize// 8) * (nsize // 8), 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, nout),
            #nn.Sigmoid(),
        )

    def forward(self, input):
       
        output = self.main(input)
        output = self.fc(output.view(-1, self.ndf*4*(self.nsize//8)*(self.nsize//8)))

        return output.view(-1, 1).squeeze(1)

def main():
    args = parse_args()
    print(args)

    vis = visdom.Visdom()


    # Initialize parameters and other flags
    c = 0.01
    n_critic = 3
    
    if args.dataset == 'MNIST' or args.dataset=='FashionMNIST':
        # Dataset and dataloader 
        dset = datasets.MNIST(args.dataroot, train=True, 
                            download=True, 
                            transform=transforms.Compose([transforms.ToTensor()]))
        nc = 1
    elif args.dataset == 'cifar10':
        dset = datasets.CIFAR10(root=args.dataroot, download=True,
                           transform=transforms.Compose([
                               transforms.Scale(64),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (.5, .5, .5)),
                           ]))
        nc = 3
    elif args.dataset == 'images':
        dset = datasets.ImageFolder(args.dataroot,
                               transform=transforms.Compose([
                                   transforms.Scale(128),
                                   transforms.CenterCrop(128),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
        nc = 3
    elif args.dataset == 'celeba':
        dset = datasets.ImageFolder('../../../datasets/celebA/',
                               transform=transforms.Compose([
                                   transforms.Scale(64),
                                   transforms.CenterCrop(64),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (.5, .5, .5)),
                               ]))
        nc = 3

    data_loader = DataLoader(dset, batch_size=args.batch_size, shuffle=True)

    # create generator and discriminator
    # create generator and discriminator
    G = _G(args.ngpu,args.nz, args.ngf, nc, nsize=64)
    G.apply(weights_init)
    
    
    D = _D(args.ngpu, args.ndf, nc, nout=1, nsize=64)
    D.apply(weights_init)
    # optimizers
    G_optim = optim.Adam(G.parameters(), lr=args.lr)
    D_optim = optim.Adam(D.parameters(), lr=args.lr)

    # Loss 
    fn_loss = nn.MSELoss()

    if args.cuda:
        G.cuda()
        D.cuda()
        fn_loss = nn.MSELoss().cuda()


    # print net 
    print("Generator Net")
    print_network(G)

    print("Discriminator Net")
    print_network(D)

    
    # variable setup
    sample_z_= Variable(torch.rand((args.batch_size, args.nz)), volatile=True)
    y_real_ = Variable(torch.ones(args.batch_size, 1))
    y_fake_ = Variable(torch.zeros(args.batch_size, 1))

    if args.cuda:
        y_real_ = Variable(torch.ones(args.batch_size, 1).cuda())
        y_fake_ = Variable(torch.zeros(args.batch_size, 1).cuda())
        sample_z_= Variable(torch.rand((args.batch_size, args.nz)), volatile=True).cuda()

    

    # train loop 
    D.train()
    print("train begins")
    for epoch in range(args.epochs):
        # set G for training 
        G.train()

        start_time = time.time()
        for iter, (x_, _) in enumerate(data_loader):
            if iter == data_loader.dataset.__len__() // args.batch_size:
                break

            # sample z 
            z_ = torch.rand((args.batch_size, args.nz))

            # for gpu 
            if args.cuda:
                x_, z_ = Variable(x_.cuda()), Variable(z_.cuda())
            else:
                x_, z_ = Variable(x_), Variable(z_)

            # optimizer reset 
            D_optim.zero_grad()
            
            

            # update D 
            # D(x)
            D_real = D(x_)
            

            #D(G(x))
            G_ = G(z_)
            D_fake = D(G_)
            
            
            
            # Uncomment D_loss and its respective G_loss of your choice
            # ---------------------------------------------------------
            # from https://github.com/wiseodd/generative-models/blob/master/GAN/f_gan/f_gan_pytorch.py

            """ Total Variation """
            # D_loss = -(torch.mean(0.5 * torch.tanh(D_real)) -
            #            torch.mean(0.5 * torch.tanh(D_fake)))
            """ Forward KL """
            # D_loss = -(torch.mean(D_real) - torch.mean(torch.exp(D_fake - 1)))
            """ Reverse KL """
            D_loss = -(torch.mean(-torch.exp(D_real)) - torch.mean(-1 - D_fake))
            """ Pearson Chi-squared """
            # D_loss = -(torch.mean(D_real) - torch.mean(0.25*D_fake**2 + D_fake))
            """ Squared Hellinger """
            # D_loss = -(torch.mean(1 - torch.exp(D_real)) -
            #            torch.mean((1 - torch.exp(D_fake)) / (torch.exp(D_fake))))

            # overall D loss 
            #D_fake_loss = torch.mean(D_fake)
            #D_real_loss = - torch.mean(D_real)
            #D_loss = D_real_loss + D_fake_loss

            # backprop D 
            D_loss.backward()

            # update weights 
            D_optim.step()

            for p in D.parameters():
                p.data.clamp_(-c, c)

            if ((iter+1) % n_critic) == 0:
                # update G 
                G_optim.zero_grad()
                G_ = G(z_)
                D_fake = D(G_)
                # from https://github.com/wiseodd/generative-models/blob/master/GAN/f_gan/f_gan_pytorch.py
                """ Total Variation """
                # G_loss = -torch.mean(0.5 * torch.tanh(D_fake))
                """ Forward KL """
                # G_loss = -torch.mean(torch.exp(D_fake - 1))
                """ Reverse KL """
                G_loss = -torch.mean(-1 - D_fake)
                """ Pearson Chi-squared """
                # G_loss = -torch.mean(0.25*D_fake**2 + D_fake)
                """ Squared Hellinger """
                # G_loss = -torch.mean((1 - torch.exp(D_fake)) / (torch.exp(D_fake)))
                #G_loss = -torch.mean(D_fake)
                # G_loss = fn_loss(D_fake, y_real_)

                # backprop G 
                G_loss.backward()
                G_optim.step()


            if (iter+1) % args.log_interval == 0:
                print("Epoch: [%2d] [%4d/%4d] D_loss: %.8f, G_loss: %.8f" %
                              ((epoch + 1), (iter + 1), 
                                data_loader.dataset.__len__() // args.batch_size, 
                                D_loss.data[0], G_loss.data[0]))
                
                vutils.save_image(x_.cpu().data,
                        '%s/wgan_real_samples.png' % args.result_dir,
                        normalize=True)
                fake = G(sample_z_)
                grid = vutils.make_grid(fake.cpu().data)
                ndarr = grid.mul(255).clamp(0, 255).byte().numpy()
                
                vis.image(ndarr)
                vutils.save_image(fake.cpu().data,
                        '%s/wgan_fs_%s_epoch_%s_ %s_%03d.png' % (args.result_dir,
                                                                args.dataset,
                                                                    args.nz, 
                                                                    epoch, iter),normalize=True)


if __name__ == '__main__':
    main()
