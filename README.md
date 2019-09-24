# pytorch-gan
Modified and example codes of GAN in pytorch. Parts of the orginal code from pytorch/examples are modified for easier experimentation.

## Dependencies
- pytorch>=0.4.0
- torchvision
- visdom
- numpy

## How to run?
In a new terminal window launch visdom server as 
`python -m visdom.server`

### DCGAN
To run dcgan on cifar10 dataset
`python dcgan.py --dataset cifar10 --dataroot ./data/cifar10`


## Colab
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ResByte/pytorch-gan/dcgan_notebook.ipynb)
