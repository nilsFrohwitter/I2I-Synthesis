"""
This code is highly related to the PyTorch-CycleGAN implementaion
from Zhu et al.:
https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix

"""

from torch.optim import lr_scheduler
import torch
import os
from visualization import visualize
from collections import OrderedDict
from savefile import save_state, load_state_test, save_as_nifti, save_images
from datasets.dataset_utils import transform_to_HU
from torch.nn import functional as nnf
import torchvision.utils as utils
import torch.nn as nn


class BaseModel():
    """Basic model for setting up basic functions
    """

    def __init__(self, opt):
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        self.optimizers = []
        self.losses = {'loss': 0}
        self.losses_sum = {'loss': 0}
        self.L1_loss = nn.L1Loss()

    def set_scheduler(self, opt):
        self.schedulers = [get_scheduler(optimizer, opt) for optimizer in self.optimizers]

    def get_current_losses(self):
        """Return training losses / errors. train.py will print out these errors on console, and save them to a file"""
        losses = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                losses[name] = float(getattr(self, name))  # float(...) works for both scalar tensor and float number
        return losses

    def update_learning_rate(self):
        """Update learning rates for all the networks; called at the end of every epoch"""
        for scheduler in self.schedulers:
            scheduler.step()

        lr = self.optimizers[0].param_groups[0]['lr']
        # print('learning rate = %.7f' % lr)  # lr is already printed elsewhere
 
    def save_model(self, epoch):
        save_state(models=self.models, optimizers=self.optimizer, losses=self.losses, epoch=epoch, opt=self.opt)
        pass

    def load_model(self):
        self.models = load_state_test(self.models, load_dir=self.opt.checkpoint_dir)

    def update_loss_sum(self):
        """Summing up the loss values of each iteration to calculate an
        epoch-wise average"""
        with torch.no_grad():
            for key in self.losses.keys():
                self.losses_sum[key] += self.losses[key]

    def add_losses_to_writer(self, writer, dataset_size, epoch):
        """Calculate loss averages and add it to the writer"""
        with torch.no_grad():
            avg_total = 0
            for key in self.losses_sum.keys():
                avg = self.losses_sum[key]/dataset_size
                writer.add_scalar(os.path.join('Loss', key.rsplit(sep='loss_')[-1]), avg, epoch)
                if key in self.primary_losses:
                    print('average ' + key + ' is %f' % avg)
                    avg_total += avg
            avg_total = avg_total / 3
            writer.add_scalar('Loss/total', avg_total, epoch)
            print('average total_loss is %f' % avg_total)

    def visualize(self, epoch=None):
        if self.opt.do_visualize:
            visualize(self.real_A, self.fake_B, self.real_B, self.fake_A, self.cycled_A, self.cycled_B, epoch=epoch)

    def visualize_diff(self, epoch=None):
        if self.opt.do_visualize:
            visualize(self.real_A, self.fake_B, self.real_B, self.fake_A, epoch=epoch)

    def save_writer_diff_images(self, writer, epoch=None):
        diff_a = torch.abs(self.real_A - self.fake_A)
        diff_b = torch.abs(self.real_B - self.fake_B)
        writer.add_image('diff/real, fake and diff images',
                         utils.make_grid([self.real_A.squeeze(0), self.fake_A.squeeze(0), diff_a.squeeze(0),
                                          self.real_B.squeeze(0), self.fake_B.squeeze(0), diff_b.squeeze(0)], nrow=3, scale_each=True),
                         epoch)

    def save_writer_cycled_images(self, writer, epoch=None):
        writer.add_image('cycled/real, fake and cycled images',
                         utils.make_grid([self.real_A.squeeze(0), self.fake_A.squeeze(0), self.cycled_A.squeeze(0),
                                          self.real_B.squeeze(0), self.fake_B.squeeze(0), self.cycled_B.squeeze(0)],
                                         nrow=3, scale_each=True),
                         epoch)

    def save_nifti_images(self, path, epoch, names=None):
        names = self.opt.val_names
        dir_image = getattr(self, 'image_' + path)
        if names[0] in dir_image[0]:
            name = names[0]
        elif names[1] in dir_image[0]:
            name = names[1]
        elif names[2] in dir_image[0]:
            name = names[2]
        else:
            print('given names should always match')
            return
        if path == 'A_paths':
            save_as_nifti(self.fake_B, self.cycled_A, self.opt,
                          dir_image[0], epoch, name=name, real_dom=self.opt.data_a, fake_dom=self.opt.data_b)
        else:
            save_as_nifti(self.fake_A, self.cycled_B, self.opt,
                          dir_image[0], epoch, name=name, real_dom=self.opt.data_b, fake_dom=self.opt.data_a)

    def save_nifti_oneimage(self, path, epoch):
        dir_image = getattr(self, 'image_' + path)
        if path == 'A_paths':
            save_as_nifti(self.fake_B.cpu(), self.cycled_A.cpu(), self.opt,
                          dir_image[0], epoch, name='test', real_dom=self.opt.data_a, fake_dom=self.opt.data_b)
        else:
            save_as_nifti(self.fake_A.cpu(), self.cycled_B.cpu(), self.opt,
                          dir_image[0], epoch, name='test', real_dom=self.opt.data_b, fake_dom=self.opt.data_a)

    def save_images(self, a_path, b_path, epoch):
        if self.opt.phase == 'test':
            state = 'test'
        else:
            state = 'val'
        save_images(self.real_A, self.real_B, a_path, b_path, epoch, state, 'real', self.opt)
        save_images(self.fake_A, self.fake_B, a_path, b_path, epoch, state, 'fake', self.opt)
        save_images(self.cycled_A, self.cycled_B, a_path, b_path, epoch, state, 'cycled', self.opt)

    def pixel_error(self, direction):
        if direction == 'AB':
            fake_B = transform_to_HU(self.fake_B, self.opt)
            real_B = transform_to_HU(self.real_B, self.opt)
            return self.L1_loss(fake_B, real_B)    # L1-loss
        elif direction == 'BA':
            # transform to HU could be unnecessary (for CT-CBCT) because both are in CBCT domain (not CT)
            fake_A = transform_to_HU(self.fake_A, self.opt)
            real_A = transform_to_HU(self.real_A, self.opt)
            return self.L1_loss(fake_A, real_A)    # L1-loss
        else:
            print('direction of pixel_error should be \'AB\' or \'BA\'')

    def split_in_patches(self):
        """splits the windowed image of size [C, B, dim1, dim2] into patches of size [crop, crop]=[256,256].
        The used data is therefore [C, B*n_patches, 256, 256]"""
        if list(self.real_A.shape[2:]) == [self.opt.crop_size, self.opt.crop_size]:
            return None     # if the image to be split is already in crop_size, do nothing
        elif (self.real_A.shape[2] <= self.opt.crop_size) or (self.real_A.shape[3] <= self.opt.crop_size):
            return None
        dh = self.opt.center_crop_height - self.opt.crop_size
        dw = self.opt.center_crop_width - self.opt.crop_size
        patches_a = self.real_A.squeeze(0).unfold(1, self.opt.crop_size, dh).unfold(2, self.opt.crop_size, dw)
        self.real_A = patches_a.contiguous().view(-1, patches_a.size(0), self.opt.crop_size, self.opt.crop_size)
        patches_b = self.real_B.squeeze(0).unfold(1, self.opt.crop_size, dh).unfold(2, self.opt.crop_size, dw)
        self.real_B = patches_b.contiguous().view(-1, patches_b.size(0), self.opt.crop_size, self.opt.crop_size)
        return True

    def fuse_patches(self, is_splitted):
        """redoes the splitting into patches. Combines the overlapping patches by calculating
        the mean values in the overlap region and creates the original size of the image.
        """
        if is_splitted:
            self.real_A = fuse_patches_real(self.real_A, self.opt)
            self.real_B = fuse_patches_real(self.real_B, self.opt)
            self.fake_A = fuse_patches_real(self.fake_A, self.opt)
            self.fake_B = fuse_patches_real(self.fake_B, self.opt)
            self.cycled_A = fuse_patches_real(self.cycled_A, self.opt)
            self.cycled_B = fuse_patches_real(self.cycled_B, self.opt)

    def get_images(self):
        return self.fake_B, self.cycled_A, self.fake_A, self.cycled_B

    def add_histcount(self, histograms):
        min_val = (self.opt.clip - self.opt.clip_range) + self.opt.hist_cut
        max_val = self.opt.clip - self.opt.hist_cut
        histograms['real_A'] += torch.histc(transform_to_HU(self.real_A, self.opt).cpu(), bins=self.opt.n_hist_bins, min=min_val, max=max_val)
        histograms['real_B'] += torch.histc(transform_to_HU(self.real_B, self.opt).cpu(), bins=self.opt.n_hist_bins, min=min_val, max=max_val)
        histograms['fake_A'] += torch.histc(transform_to_HU(self.fake_A, self.opt).cpu(), bins=self.opt.n_hist_bins, min=min_val, max=max_val)
        histograms['fake_B'] += torch.histc(transform_to_HU(self.fake_B, self.opt).cpu(), bins=self.opt.n_hist_bins, min=min_val, max=max_val)
        histograms['cycled_A'] += torch.histc(transform_to_HU(self.cycled_A, self.opt).cpu(), bins=self.opt.n_hist_bins, min=min_val, max=max_val)
        histograms['cycled_B'] += torch.histc(transform_to_HU(self.cycled_B, self.opt).cpu(), bins=self.opt.n_hist_bins, min=min_val, max=max_val)
        return histograms


def fuse_patches_real(patches, opt):
    # bringing the data in shape for fold
    uf = patches.contiguous().view(opt.batch_size, opt.output_nc, -1, opt.crop_size ** 2)
    uf = uf.permute(0, 1, 3, 2)
    uf = uf.contiguous().view(opt.batch_size, opt.output_nc * (opt.crop_size ** 2), -1)
    # folds the patches back together with summed overlap
    raw = nnf.fold(uf, output_size=(opt.center_crop_height, opt.center_crop_width),
                   kernel_size=(opt.crop_size, opt.crop_size),
                   stride=(opt.center_crop_height - opt.crop_size, opt.center_crop_width - opt.crop_size))
    # counter for number of patches overlapping in the pixels of the big image
    counter = nnf.fold(torch.ones_like(uf), output_size=(opt.center_crop_height, opt.center_crop_width),
                       kernel_size=(opt.crop_size, opt.crop_size),
                       stride=(opt.center_crop_height - opt.crop_size, opt.center_crop_width - opt.crop_size))
    real = raw / counter    # returns the mean in overlapping regions
    return real


def get_scheduler(optimizer, opt):
    """Returns a learning rate scheduler


    We keep the same learning rate for the first <n_epochs> epochs
    and then linearly decay the rate to zero
    The default value of n_epochs is 100. (Default max epochs: 200)
    """
    def lambda_rule(epoch):
        lr_1 = 1.0 - max(0, epoch + 1 - opt.n_epochs) / float(opt.n_epochs_decay)
        return lr_1

    return lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)


def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
    Parameters:
        nets (network list)   -- a list of networks
        requires_grad (bool)  -- whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad
