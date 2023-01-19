"""
MIT License

Copyright (c) 2021 Junyu Chen

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE."""

import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F


class MutualInformation(nn.Module):
    """
    Mutual Information
    """

    def __init__(self, sigma_ratio=1, minval=0., maxval=1., num_bin=32):
        # modified minval from 0 to -1
        super(MutualInformation, self).__init__()

        """Create bin centers"""
        bin_centers = np.linspace(minval, maxval, num=num_bin)
        vol_bin_centers = Variable(torch.linspace(minval, maxval, num_bin), requires_grad=False).cuda()
        num_bins = len(bin_centers)

        """Sigma for Gaussian approx."""
        sigma = np.mean(np.diff(bin_centers)) * sigma_ratio
        print(sigma)

        self.preterm = 1 / (2 * sigma**2)
        self.bin_centers = bin_centers
        self.max_clip = maxval
        # self.min_clip = minval      # modified: new variable
        self.num_bins = num_bins
        self.vol_bin_centers = vol_bin_centers

    def mi(self, y_true, y_pred):
        # changes ...
        y_true = y_true / 2 + 0.5
        y_pred = y_pred / 2 + 0.5

        y_pred = torch.clamp(y_pred, 0, self.max_clip)  # mod. 0 to min_clip
        y_true = torch.clamp(y_true, 0, self.max_clip)  # mod. 0 to min_clip

        y_true = y_true.view(y_true.shape[0], -1)
        y_true = torch.unsqueeze(y_true, 2)
        y_pred = y_pred.view(y_pred.shape[0], -1)
        y_pred = torch.unsqueeze(y_pred, 2)

        nb_voxels = y_pred.shape[1]  # total num of voxels

        """Reshape bin centers"""
        o = [1, 1, np.prod(self.vol_bin_centers.shape)]
        vbc = torch.reshape(self.vol_bin_centers, o).cuda()

        """compute image terms by approx. Gaussian dist."""
        I_a = torch.exp(- self.preterm * torch.square(y_true - vbc))
        I_a = I_a / torch.sum(I_a, dim=-1, keepdim=True)

        I_b = torch.exp(- self.preterm * torch.square(y_pred - vbc))
        I_b = I_b / torch.sum(I_b, dim=-1, keepdim=True)

        # compute probabilities
        pab = torch.bmm(I_a.permute(0, 2, 1), I_b)
        pab = pab/nb_voxels
        pa = torch.mean(I_a, dim=1, keepdim=True)
        pb = torch.mean(I_b, dim=1, keepdim=True)

        papb = torch.bmm(pa.permute(0, 2, 1), pb) + 1e-6
        mi = torch.sum(torch.sum(pab * torch.log(pab / papb + 1e-6), dim=1), dim=1)
        return mi.mean()  # average across batch

    def forward(self, y_true, y_pred):
        return -self.mi(y_true, y_pred)


class MutualInformation2(nn.Module):
    """
    Mutual Information Version 2
    """

    def __init__(self, num_bin=32):
        super(MutualInformation2, self).__init__()
        self.num_bins = num_bin

    def mi(self, y_true, y_pred):
        # changes ...
        y_true = y_true / 2 + 0.5
        y_pred = y_pred / 2 + 0.5

        hist2D, x_edges, y_edges = np.histogram2d(y_true.ravel())

        y_pred = torch.clamp(y_pred, 0, self.max_clip)  # mod. 0 to min_clip
        y_true = torch.clamp(y_true, 0, self.max_clip)  # mod. 0 to min_clip

        y_true = y_true.view(y_true.shape[0], -1)
        y_true = torch.unsqueeze(y_true, 2)
        y_pred = y_pred.view(y_pred.shape[0], -1)
        y_pred = torch.unsqueeze(y_pred, 2)

        nb_voxels = y_pred.shape[1]  # total num of voxels

        """Reshape bin centers"""
        o = [1, 1, np.prod(self.vol_bin_centers.shape)]
        vbc = torch.reshape(self.vol_bin_centers, o).cuda()

        """compute image terms by approx. Gaussian dist."""
        I_a = torch.exp(- self.preterm * torch.square(y_true - vbc))
        I_a = I_a / torch.sum(I_a, dim=-1, keepdim=True)

        I_b = torch.exp(- self.preterm * torch.square(y_pred - vbc))
        I_b = I_b / torch.sum(I_b, dim=-1, keepdim=True)

        # compute probabilities
        pab = torch.bmm(I_a.permute(0, 2, 1), I_b)
        pab = pab/nb_voxels
        pa = torch.mean(I_a, dim=1, keepdim=True)
        pb = torch.mean(I_b, dim=1, keepdim=True)

        papb = torch.bmm(pa.permute(0, 2, 1), pb) + 1e-6
        mi = torch.sum(torch.sum(pab * torch.log(pab / papb + 1e-6), dim=1), dim=1)
        return mi.mean()  # average across batch

    def forward(self, y_true, y_pred):
        return -self.mi(y_true, y_pred)


def criterion_GAN(pred, target_is_real, use_sigmoid=True):
    if use_sigmoid:
        if target_is_real:
            target_var = Variable(pred.data.new(pred.size()).long().fill_(1.))
        else:
            target_var = Variable(pred.data.new(pred.size()).long().fill_(0.))

        loss = F.binary_cross_entropy(pred, target_var)
    else:
        if target_is_real:
            target_var = Variable(pred.data.new(pred.size()).fill_(1.))
        else:
            target_var = Variable(pred.data.new(pred.size()).fill_(0.))

        loss = F.mse_loss(pred, target_var)

    return loss
