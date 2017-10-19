import torch, time, os, pickle
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision.utils as vutils

# as the name says
def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)

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

class generator(nn.Module):
    # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
    # Architecture : FC1024_BR-FC7x7x128_BR-(64)4dc2s_BR-(1)4dc2s_S
    def __init__(self, dataset = 'mnist'):
        super(generator, self).__init__()
        if dataset == 'mnist' or dataset == 'fashion-mnist':
            self.input_height = 28
            self.input_width = 28
            self.input_dim = 62
            self.output_dim = 1
        elif dataset == 'celebA':
            self.input_height = 64
            self.input_width = 64
            self.input_dim = 62
            self.output_dim = 3

        self.fc = nn.Sequential(
            nn.Linear(self.input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 128 * (self.input_height // 4) * (self.input_width // 4)),
            nn.BatchNorm1d(128 * (self.input_height // 4) * (self.input_width // 4)),
            nn.ReLU(),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, self.output_dim, 4, 2, 1),
            nn.Sigmoid(),
        )
        initialize_weights(self)

    def forward(self, input):
        x = self.fc(input)
        x = x.view(-1, 128, (self.input_height // 4), (self.input_width // 4))
        x = self.deconv(x)

        return x


class discriminator(nn.Module):
    # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
    # Architecture : (64)4c2s-(128)4c2s_BL-FC1024_BL-FC1_S
    def __init__(self, dataset = 'mnist'):
        super(discriminator, self).__init__()
        if dataset == 'mnist' or dataset == 'fashion-mnist':
            self.input_height = 28
            self.input_width = 28
            self.input_dim = 1
            self.output_dim = 1
        elif dataset == 'celebA':
            self.input_height = 64
            self.input_width = 64
            self.input_dim = 3
            self.output_dim = 1

        self.conv = nn.Sequential(
            nn.Conv2d(self.input_dim, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
        )
        self.fc = nn.Sequential(
            nn.Linear(128 * (self.input_height // 4) * (self.input_width // 4), 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, self.output_dim),
            nn.Sigmoid(),
        )
        initialize_weights(self)

    def forward(self, input):
        x = self.conv(input)
        x = x.view(-1, 128 * (self.input_height // 4) * (self.input_width // 4))
        x = self.fc(x)

        return x


# Initialize parameters and other flags
BATCH_SIZE = 128
nb_hidden = 62
nb_epochs = 10
lr = 0.001
momentum = 0.5
cuda = True
log_interval = 10
dataset_name = 'mnist'
dataroot = './data'
image_h = 64
image_w = 64
kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
out_dir = './samples/'


# create generator and discriminator
G = generator()
D = discriminator()

# optimizers
G_optim = optim.Adam(G.parameters(), lr=0.001)
D_optim = optim.Adam(D.parameters(), lr=0.001)

# Loss 
fn_loss = nn.MSELoss()

if cuda:
    G.cuda()
    D.cuda()
    fn_loss = nn.MSELoss().cuda()

# print net 
print("Generator Net")
print_network(G)

print("Discriminator Net")
print_network(D)


# Dataset and dataloader 
dset = datasets.MNIST('data', train=True, 
                            download=True, 
                            transform=transforms.Compose([transforms.ToTensor()]))
data_loader = DataLoader(dset, batch_size=BATCH_SIZE, shuffle=True)

sample_z_= Variable(torch.rand((BATCH_SIZE, nb_hidden)), volatile=True)


# Training loop 
y_real_ = Variable(torch.ones(BATCH_SIZE, 1))
y_fake_ = Variable(torch.zeros(BATCH_SIZE, 1))


if cuda:
    y_real_ = Variable(torch.ones(BATCH_SIZE, 1).cuda())
    y_fake_ = Variable(torch.zeros(BATCH_SIZE, 1).cuda())

# set D for training 
D.train()
print("Train begins")
for epoch in range(nb_epochs):
    # set G for training
    G.train()
    start_time = time.time()
    for iter, (x_, _) in enumerate(data_loader):
        if iter == data_loader.dataset.__len__() //BATCH_SIZE:
            break

        # sample z 
        z_ = torch.rand((BATCH_SIZE, nb_hidden))
        
        # create both input and random noise as variable
        if cuda:
            x_, z_ = Variable(x_.cuda()), Variable(z_.cuda())
        else:
            x_, z_ = Variable(x_), Variable(z_)


        # set optimzer grads to zero 
        D_optim.zero_grad()

        # update D 
        # D(x)
        D_real = D(x_)
        D_real_loss = fn_loss(D_real,y_real_)

        # D(G(x))
        G_ = G(z_)
        D_fake = D(G_)
        D_fake_loss = fn_loss(D_fake, y_fake_)

        # overall D loss = D(x) + D(G(z))
        D_loss = D_real_loss + D_fake_loss

        # backprop D 
        D_loss.backward()
        # update weights 
        D_optim.step()

        # update G 
        G_optim.zero_grad()
        G_ = G(z_)
        D_fake = D(G_)
        G_loss = fn_loss(D_fake, y_real_)

        # backprop G 
        G_loss.backward()
        G_optim.step()


        if iter % 100 == 0:
            print("Epoch: [%2d] [%4d/%4d] D_loss: %.8f, G_loss: %.8f" %
                          ((epoch + 1), (iter + 1), data_loader.dataset.__len__() // BATCH_SIZE, D_loss.data[0], G_loss.data[0]))
            vutils.save_image(x_.cpu().data,
                    '%s/lsgan_real_samples.png' % out_dir,
                    normalize=True)
            fake = G(z_)
            vutils.save_image(fake.cpu().data,
                    '%s/lsgan_fake_samples_epoch_%03d.png' % (out_dir, epoch),
                    normalize=True)






