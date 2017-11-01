# pytorch-gan
Modified and example codes of GAN in pytorch. Parts of the orginal code from pytorch/examples are modified for easier experimentation.

## Dependencies
- pytorch>=0.2.0
- torchvision
- visdom
- numpy


## DCGAN
To run dcgan on cifar10 dataset
`python dcgan_pytorch.py --dataset cifar10 --dataroot ./data/cifar10`

## AEGAN(AutoEncoder GAN)

`python ae_gan.py --dataset cifar10 --dataroot ./data/cifar10 --cuda --batchSize 16`
