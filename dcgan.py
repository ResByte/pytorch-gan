import argparse
import os 
import random
import torch 
import torch.nn as nn 
import utils  
from  networks import Generator, Disciminator 
from tensorboardX import SummaryWriter
import visdom
import torchvision.utils as vutils

def main():
    # load config
    opt = utils.config()
    device = torch.device("cuda:0" if opt.cuda else "cpu")

    # load dataset 
    dataset = utils.get_dataset(opt)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                            shuffle=True, num_workers=int(opt.workers))

    # params 
    nz = int(opt.nz)  # z dimensions
    ngf = int(opt.ngf)   # generator filters
    ndf = int(opt.ndf)   # discriminator filters
    nc = 3               # image channels 

    # generator
    G = Generator(nz,ngf,nc).to(device)
    G.apply(utils.weights_init)

    # discriminator
    D = Disciminator(nc, ndf).to(device)
    D.apply(utils.weights_init)

    #optimizer
    optimD = torch.optim.Adam(D.parameters(), 
                                lr=opt.lr, betas=(opt.beta1, 0.999))
    optimG = torch.optim.Adam(G.parameters(),
                                lr=opt.lr, betas=(opt.beta1, 0.999))
    
    # loss function
    criterion = nn.BCELoss()

    # fixed variable 
    fixed_noise = torch.FloatTensor(opt.batchSize, nz, 1, 1).normal_(0,1).to(device)

    vis = visdom.Visdom()
    lot = vis.line(
                X=torch.zeros((1,)).cpu(),
                Y=torch.zeros((1,2)).cpu(),
                opts=dict(
                    xlabel='Iterations',
                    ylabel='Loss',
                    title='Current Losses',
                    legend=['Gen Loss', 'Disc Loss'])
                )
    
    # training loop 
    count = 0
    for epoch in range(opt.niter):
        for i, (inputs, targets) in enumerate(dataloader):
            count+=1
            inputs, targets = inputs.to(device), targets.to(device)
            batch_size = inputs.size(0)
            
            zeros = torch.zeros(batch_size).to(device)
            ones = torch.ones(batch_size).to(device)

            # update D network 
            # maximize log(D(x)) + log(1 - D(G(z)))
            optimD.zero_grad()

            # train with real : log(D(x))
            d_real = D(inputs)
            # distance from true real
            errD_real = criterion(d_real,ones)
            errD_real.backward()                  # gradient for D only
            D_x = d_real.data.mean()

            # train with fake : log(1-D(G(z))) 
            # create noise 
            mini_batch_noise = torch.randn((batch_size, nz, 1, 1)).float().to(device)
            # generate image 
            g_fake = G(mini_batch_noise).detach() # to remove gradient update
            d_fake = D(g_fake)
            errD_fake = criterion(d_fake, zeros)
            errD_fake.backward()                  # gradient for D only
            D_G_z1 = d_fake.data.mean()
            
            total_errD = errD_real + errD_fake
            optimD.step()

            # update G network 
            # maximize log(D(G(z)))
            optimG.zero_grad()

            # sample new size 
            # create minibatch noise 
            mini_batch_noise = torch.randn((batch_size, nz, 1, 1)).float().to(device)

            g_fake = G(mini_batch_noise)           # create image
            output = D(g_fake)                     
            errG = criterion(output, ones)         # make generated image to real
            errG.backward()
            D_G_z2 = output.data.mean()
            optimG.step()

            print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
              % (epoch, opt.niter, i, len(dataloader),
                 total_errD.data[0], errG.data[0], D_x, D_G_z1, D_G_z2))
            vis.line(
                        X=torch.ones((1, 2)).cpu()*count,
                        Y=torch.Tensor([errG.data[0],total_errD.data[0]]).unsqueeze(0).cpu(),
                        win=lot,
                        update='append'
                    )
            if i % 100 == 0:
                # vutils.save_image(real_cpu,
                #         '%s/real_samples.png' % opt.outf,
                #         normalize=True)
                fake = G(fixed_noise)
                
                grid = vutils.make_grid(fake.cpu().data)
                ndarr = grid.mul(255).clamp(0, 255).byte().numpy()

                vis.image(ndarr)
                # vutils.save_image(fake.data,
                #         '%s/fake_samples_epoch_%03d.png' % (opt.outf, epoch),
                #         normalize=True)

        # if do checkpointing
        # torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (opt.outf, epoch))
        # torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (opt.outf, epoch))





    


if __name__ == "__main__":
    main()
    