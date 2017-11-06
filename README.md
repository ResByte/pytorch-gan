# pytorch-gan
Modified and example codes of GAN in pytorch. Parts of the orginal code from pytorch/examples are modified for easier experimentation.

## Dependencies
- pytorch>=0.2.0
- torchvision
- visdom
- numpy

## How to run?
In a new terminal window launch visdom server as 
`python -m visdom.server`

### DCGAN
To run dcgan on cifar10 dataset
`python dcgan_pytorch.py --dataset cifar10 --dataroot ./data/cifar10`

### AEGAN(AutoEncoder GAN)

`python aegan.py --dataset cifar10 --dataroot ./data/cifar10 --cuda --batchSize 16`


### Vanilla AutoEncoder(AE)
`python autoencoder.py --dataset cifar10 --dataroot ./data/cifar10 --cuda --batchSize 16`
