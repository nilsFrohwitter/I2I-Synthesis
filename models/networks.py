"""
This code is highly related to the PyTorch-CycleGAN implementaion
from Zhu et al.:
https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix

"""

import torch.nn as nn
from torch.nn import init
import numpy as np
from models.modules import get_norm_layer, ResidualBlock, ConvStride2Block, ConvTransposeBlock, ConvBlock



class BaseNetwork(nn.Module):

    def weights_init(self, name, init_type='normal', init_gain=0.02, is_netE=False):
        def init_func(m):
            classname = m.__class__.__name__
            # for every Linear layer in a model..
            if hasattr(m, 'weight') and (classname.find('Linear') != -1 or classname.find('Conv') != -1):
                # apply a uniform distribution to the weights and a bias=0
                if init_type == 'normal':
                    init.normal_(m.weight.data, 0.0, init_gain)
                else:
                    raise NotImplementedError('initialization method [%s] is not implemented' % init_type)

                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)
            elif classname.find('BatchNorm2d') != -1:
                init.normal_(m.weight.data, 1.0, init_gain)
                init.constant_(m.bias.data, 0.0)

        n_parameter = 0
        for parameter in self.model.parameters():
            n_parameter += np.prod(parameter.size())
        self.model.apply(init_func)
        print('initialized network %s with %d parameters' % (name, n_parameter))

    def init_net(self, opt, name='noName'):
        if len(opt.gpu_ids) > 0:
            self.model.to(opt.gpu_ids[0])
            self.model = nn.DataParallel(self.model, opt.gpu_ids)
            self.weights_init(name, init_type=opt.init_type, init_gain=opt.init_gain)


class Generator(BaseNetwork):

    def __init__(self, opt):
        """
            ngf: number of generator features
        """
        super(Generator, self).__init__()
        # new: use instancenorm and bias=True
        ngf = opt.ngf
        norm = opt.norm
        self.model = nn.Sequential(
            ConvBlock(opt.input_nc, ngf, 'front', norm),
            ConvStride2Block(ngf, ngf * 2, norm),
            ConvStride2Block(ngf * 2, ngf * 4, norm),
            ResidualBlock(ngf * 4, ngf * 4, norm),
            ResidualBlock(ngf * 4, ngf * 4, norm),
            ResidualBlock(ngf * 4, ngf * 4, norm),
            ResidualBlock(ngf * 4, ngf * 4, norm),
            ResidualBlock(ngf * 4, ngf * 4, norm),
            ResidualBlock(ngf * 4, ngf * 4, norm),
            ResidualBlock(ngf * 4, ngf * 4, norm),
            ResidualBlock(ngf * 4, ngf * 4, norm),
            ResidualBlock(ngf * 4, ngf * 4, norm),
            ConvTransposeBlock(ngf * 4, ngf * 2, norm),
            ConvTransposeBlock(ngf * 2, ngf, norm),
            ConvBlock(ngf, opt.output_nc, 'back', norm)
        )

    def forward(self, x):
        return self.model(x)


class Discriminator(BaseNetwork):
    r"""70x70 PatchGAN from the implementation of CycleGAN in torch.

    Every pixel in the last layer represents a 70x70 patch of the input image.
    """

    def __init__(self, opt, use_sigmoid=False):
        super(Discriminator, self).__init__()
        ndf = opt.ndf
        self.norm_layer, self.use_bias = get_norm_layer(opt.norm)

        sequence = [
            # input is (nc) x 256 x 256
            nn.Conv2d(opt.output_nc, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),
            # size: (ndf) x 128 x 128
            nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1, bias=self.use_bias),
            self.norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            # size: (ndf*2) x 64 x 64
            nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1, bias=self.use_bias),
            self.norm_layer(ndf * 4),
            nn.LeakyReLU(0.2, True),
            # size: (ndf*4) x 32 x 32
            nn.Conv2d(ndf * 4, ndf * 4, kernel_size=4, stride=1, padding=1, bias=self.use_bias),
            self.norm_layer(ndf * 4),
            nn.LeakyReLU(0.2, True),
            # size: (ndf*4) x 31 x 31
            nn.Conv2d(ndf * 4, 1, kernel_size=4, stride=1, padding=1)
            # size: (1) x 30 x 30
        ]
        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, x):
        return self.model(x)
