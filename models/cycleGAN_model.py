"""
This code is highly related to the PyTorch-CycleGAN implementaion
from Zhu et al.:
https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix

"""
import torch
import copy
import itertools
import torch.nn.functional as F
import torch.nn as nn
from .networks import Generator, Discriminator
from .additional_losses import MutualInformation
from .base_model import BaseModel, set_requires_grad
from utils.image_pool import ImagePool


class CycleGANModel(BaseModel):
    """

    """

    def __init__(self, opt):
        super().__init__(opt)
        self.losses = {'loss_D_A': 0, 'loss_D_B': 0, 'loss_G': 0,
                       'loss_idt_A': 0, 'loss_idt_B': 0, 'loss_G_A': 0, 'loss_G_B': 0,
                       'loss_cycle_A': 0, 'loss_cycle_B': 0, 'loss_G_grad': 0,
                       'loss_MI_A': 0, 'loss_MI_B': 0, 'loss_MI': 0}
        self.losses_sum = copy.deepcopy(self.losses)
        self.primary_losses = ['loss_D_A', 'loss_D_B', 'loss_G_A', 'loss_G_B', 'loss_G']
        self.optimizers = []
        self.image_paths = []

        # create the Generators
        self.netG_A = Generator(opt)
        self.netG_A.init_net(opt, name='G_A')
        self.netG_B = Generator(opt)
        self.netG_B.init_net(opt, name='G_B')
        self.models = {'G_A': self.netG_A, 'G_B': self.netG_B}
        self.sobel_x = sobel('x', opt.output_nc, opt.input_nc).to(self.device)
        self.sobel_y = sobel('y', opt.output_nc, opt.input_nc).to(self.device)

        # create Discriminator and Optimizer when training
        if self.isTrain:
            self.netD_A = Discriminator(opt)
            self.netD_A.init_net(opt, name='D_A')
            self.netD_B = Discriminator(opt)
            self.netD_B.init_net(opt, name='D_B')
            self.models.update({'D_A': self.netD_A, 'D_B': self.netD_B})

        if self.isTrain:
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers = [self.optimizer_G, self.optimizer_D]
            self.optimizer = {'optim_G': self.optimizer_G, 'optim_D': self.optimizer_D}

            self.criteriumGAN = nn.MSELoss().to(self.device)
            self.criteriumCycle = nn.L1Loss()
            self.criteriumIdt = nn.L1Loss()
            self.criteriumMI = MutualInformation()

            self.fake_B_pool = ImagePool(opt.pool_size)
            self.fake_A_pool = ImagePool(opt.pool_size)

    def set_input(self, data):
        self.real_A = data['A'].to(self.device)
        self.real_B = data['B'].to(self.device)
        self.image_A_paths = data['A_paths']
        self.image_B_paths = data['B_paths']

    def forward(self):
        """forward computes all generated fake images and reconstructed cycled images
        """
        self.fake_B = self.netG_A(self.real_A)
        self.cycled_A = self.netG_B(self.fake_B)
        self.fake_A = self.netG_B(self.real_B)
        self.cycled_B = self.netG_A(self.fake_A)

    def calc_loss_D(self, netD, fake, real):
        disc_real = netD(real)
        disc_fake = netD(fake.detach())  # netD_A(fake_A)

        # real
        loss_D_real = self.criteriumGAN(disc_real, torch.ones(disc_fake.size()).to(self.device))
        # fake
        loss_D_fake = self.criteriumGAN(disc_fake, torch.zeros(disc_fake.size()).to(self.device))
        # combined loss
        loss_D = (loss_D_real + loss_D_fake) * 0.5 * self.opt.lambda_D
        return loss_D

    def calc_loss_D_train(self):
        """use image_pool for more variaty in image pairs while training"""
        self.fake_A = self.fake_A_pool.query(self.fake_A)
        self.fake_B = self.fake_B_pool.query(self.fake_B)

        self.losses['loss_D_A'] = self.calc_loss_D(self.netD_A, self.fake_A, self.real_A)
        self.losses['loss_D_B'] = self.calc_loss_D(self.netD_B, self.fake_B, self.real_B)

    def calc_loss_D_val(self):
        """Image pool is not for validation. Don't mix it up"""
        self.losses['loss_D_A'] = self.calc_loss_D(self.netD_A, self.fake_A, self.real_A)
        self.losses['loss_D_B'] = self.calc_loss_D(self.netD_B, self.fake_B, self.real_B)

    def calc_loss_G(self):
        lambda_GAN = self.opt.lambda_GAN
        lambda_idt = self.opt.lambda_idt
        lambda_grad = self.opt.lambda_grad
        lambda_MI_A = self.opt.lambda_MI_A
        lambda_MI_B = self.opt.lambda_MI_B

        # identity loss
        self.idt_b = self.netG_B(self.real_A)
        self.idt_a = self.netG_A(self.real_B)
        self.losses['loss_idt_A'] = self.criteriumIdt(self.real_B, self.idt_a) * lambda_GAN * lambda_idt
        self.losses['loss_idt_B'] = self.criteriumIdt(self.real_A, self.idt_b) * lambda_GAN * lambda_idt

        # GAN loss
        self.disc_fake_B = self.netD_B(self.fake_B)
        self.disc_fake_A = self.netD_A(self.fake_A)
        self.losses['loss_G_A'] = self.criteriumGAN(self.disc_fake_B, torch.ones(self.disc_fake_B.size()).to(self.device))
        self.losses['loss_G_B'] = self.criteriumGAN(self.disc_fake_A, torch.ones(self.disc_fake_B.size()).to(self.device))

        # Cycle loss
        self.losses['loss_cycle_A'] = self.criteriumCycle(self.cycled_A, self.real_A) * lambda_GAN
        self.losses['loss_cycle_B'] = self.criteriumCycle(self.cycled_B, self.real_B) * lambda_GAN

        # Gradient loss
        self.losses['loss_G_grad'] = self.calc_grad_loss() * lambda_grad
  
        # MutualInformation loss
        self.losses['loss_MI_A'] = self.criteriumMI(y_true=self.real_A, y_pred=self.fake_B)
        self.losses['loss_MI_B'] = self.criteriumMI(y_true=self.real_B, y_pred=self.fake_A)
        self.losses['loss_MI'] = self.losses['loss_MI_A'] * lambda_MI_A + self.losses['loss_MI_B'] * lambda_MI_B

        # combined loss
        self.losses['loss_G'] = 0
        self.losses['loss_G'] = self.losses['loss_G_A'] + self.losses['loss_G_B'] \
            + self.losses['loss_cycle_A'] + self.losses['loss_cycle_B'] + self.losses['loss_MI'] \
            + self.losses['loss_idt_A'] + self.losses['loss_idt_A'] + self.losses['loss_G_grad']

    def optimize_parameters(self):

        self.forward()
        # optimize G
        set_requires_grad([self.netD_A, self.netD_B], False)    # D net gradients are in this part not to be included
        self.optimizer_G.zero_grad()
        self.calc_loss_G()
        self.losses['loss_G'].backward()
        self.optimizer_G.step()
        # optimize D
        set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad()
        self.calc_loss_D_train()
        self.losses['loss_D_A'].backward()
        self.losses['loss_D_B'].backward()
        self.optimizer_D.step()

    def calc_grad_loss(self):
        self.sob_x_a = F.conv2d(self.real_A - self.fake_B, self.sobel_x, padding=1)
        self.sob_y_a = F.conv2d(self.real_A - self.fake_B, self.sobel_y, padding=1)
        self.sob_x_b = F.conv2d(self.real_B - self.fake_A, self.sobel_x, padding=1)
        self.sob_y_b = F.conv2d(self.real_B - self.fake_A, self.sobel_y, padding=1)
        sob_x_a_loss = self.criteriumCycle(self.sob_x_a, torch.zeros_like(self.real_A))
        sob_y_a_loss = self.criteriumCycle(self.sob_y_a, torch.zeros_like(self.real_A))
        sob_x_b_loss = self.criteriumCycle(self.sob_x_b, torch.zeros_like(self.real_A))
        sob_y_b_loss = self.criteriumCycle(self.sob_y_b, torch.zeros_like(self.real_A))
        return sob_x_a_loss + sob_y_a_loss + sob_x_b_loss + sob_y_b_loss

    def eval(self):
        if self.opt.eval:
            self.netD_A.eval()
            self.netD_B.eval()
            self.netG_B.eval()
            self.netG_A.eval()


def sobel(direction, nc_out, nc_in):
    sobel = [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]    # y direction
    # sobel = [[3, 10, 3], [0, 0, 0], [-3, -10, -3]]    # alternative form
    sobel_kernel = torch.FloatTensor(sobel)
    if direction == 'x':
        sobel_kernel.transpose(0, 1)
    return sobel_kernel.repeat(nc_out, nc_in, 1, 1)
