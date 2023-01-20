"""
This code is highly related to the PyTorch-CycleGAN implementaion
from Zhu et al.:
https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix

"""

import argparse
import os
import uuid
import utils.util
import torch
import datetime
import csv
import platform


class BaseOptions():
    """This class defines the options used in training and testing.

    Furthermore it provides functions for printing and saving the options.
    """

    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
        current_system = platform.system()

        # which model to use (for now, only CycleGAN)
        parser.add_argument('--model', type=str, choices=['CycleGAN'], default='CycleGAN')
        model = parser.parse_args().model
        # save parameters
        if current_system == 'Linux':
            sv_dir = '/netscratch/frohwitter/results/' + model + ' MRCT'  # e. g.: [' MRCT', ' h2z']
            parser.add_argument('--dataroot', type=str, default='/ds/images/MR_CT_LOOCV_new', help='path to images for train/val/test, e. g. /ds/images/<MR_CT_images, MR_CT_LOOCV_new>, /netscratch/frohwitter/horse2zebra_reduced')
            parser.add_argument('--dataroot_a', type=str, default=None, help='specified when different to other dataroot, otherwise set to be None; in CBCT/CT for CBCT with table and CT without')
            parser.add_argument('--dataroot_val_volume', type=str, default='/ds/images/MR_CT_LOOCV_new/val/img', help='specific dataroot for validation data volumes, e. g. /ds/images/MR_CT_LOOCV_new/val/img')
        elif current_system == 'Windows':
            sv_dir = os.path.join(r'C:\Users\Nils\Documents\Synthesis CT MR\results', model + ' h2z')
            parser.add_argument('--dataroot', type=str, default=r'C:\Users\Nils\Documents\Synthesis CT MR\Data\MR CT data\MR_CT_LOOCV', help='path to images for train/val/test')
            parser.add_argument('--dataroot_a', type=str, default=None, help='specified when different to other dataroot, otherwise set to be None; in CBCT/CT for CBCT with table and CT without')
            parser.add_argument('--dataroot_val_volume', type=str, default=r'C:\Users\Nils\Documents\Synthesis CT MR\Data\MR CT data\MR_CT_LOOCV\val\img', help='specific dataroot for validation data volumes')
        parser.add_argument('--platform', type=str, default=current_system)
        parser.add_argument('--data_a', type=str, default='MR', help='the name of the first dataset type, e. g. ["MR","pferd"]')
        parser.add_argument('--data_b', type=str, default='CT', help='the name of the second dataset type, e. g. ["CT","zebra"]')
        parser.add_argument('--img_size', type=list, default=[160, 192], help='dimension of the images used in this process, e. g. [[160, 192], [256, 256]]')
        parser.add_argument('--val_as_vol', type=str, default=True, help='to run validation volumewise, this can be set to True')
        parser.add_argument('--do_LOOCV', type=str, default=True, help='if one wants to run a Leave-One-Out-Cross-Validation for the e. g. 8 paired volumes for training: set to True')
        parser.add_argument('--val_names', type=list, default=['12', '14', '16'], help='set to one of the training volume numbers e.g. 02, 04, 06, 08, 10, 12, 14, 16. This is the suffix in the data file names.')
        parser.add_argument('--save_dir', type=str, default=sv_dir, help='directory where the run is saved')
        parser.add_argument('--name', type=str, default=datetime.datetime.now().strftime('%y-%m-%d_%H-%M-%S_') + str(uuid.uuid1()), help='name of the experiment')
        parser.add_argument('--gpu_ids', type=str, default='0', help='for example: [0,1,2]. use [-1] for CPU-mode')
        parser.add_argument('--data_format', type=str, default='nii', help='[nii | jpg | JPG | jpeg | JPEG | png] are the current usable data formats; nii is a special case')
        parser.add_argument('--seed', type=int, default=42, help='just a random seed for partial recreation, can be set on other values')
        parser.add_argument('--save_freq', type=int, default=50, help='specifies the frequency the model is saved')
        parser.add_argument('--do_visualize', type=str, default=False, help='specifies if validation results are shown in the console.')

        # for loading checkpoint
        checkpoint_name = '08-10_00-00_c00accd4-da8b-11ea-ab8f-ce679eaf1e59'
        state_checkout = 'checkpoints/epoch_20_08-10_11-52-26.pth'   # starts with checkpoints/...
        parser.add_argument('--checkpoint_dir', type=str, default=None, help='os.path.join(sv_dir, checkpoint_name, state_checkout) for example')

        # models specific parameters
        parser.add_argument('--input_nc', type=int, default=1, help='# input channels: 3 for RBG, 1 for greyscale')
        parser.add_argument('--output_nc', type=int, default=1, help='# output channels: 3 for RBG, 1 for greyscale')
        if model in ['CycleGAN']:
            parser.add_argument('--ngf', type=int, default=64, help='# gen channels after first conv-layer. original 64')
            parser.add_argument('--ndf', type=int, default=64, help='# dis channels after first conv-layer. original 64')
            parser.add_argument('--netD', type=str, default='patchGAN_70', help='specify model; default is a 70x70 PatchGAN (other patch sizes are possible, but needs to be implemented)')
            parser.add_argument('--netG', type=str, default='resnet_9blocks', help='specify model; (resnet_6blocks could be possible, but needs to be implemeted)')
            parser.add_argument('--norm', type=str, default='instance', help='[instance | batch] for normalization')
        parser.add_argument('--init_type', type=str, default='normal', help='initialize weights [normal]')
        parser.add_argument('--init_gain', type=float, default=0.02, help='scaling factor weight init')

        # other base parameters
        parser.add_argument('--num_threads', type=int, default=4, help='# threads for loading the data')
        parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
        parser.add_argument('--center_crop_height', type=int, default=480, help='height of extracted centered patch; ignored if img size is smaller.')
        parser.add_argument('--center_crop_width', type=int, default=480, help='width of extracted centered patch; ignored if img size is smaller.')
        parser.add_argument('--clip', type=int, default=600, help='max intensity value where the image values are clipped (e. g. CT images with values ranging [-1000, 1700])')
        parser.add_argument('--clip_range', type=int, default=1000, help='range of clip resulting in [clip-clip_range, clip]')
        parser.add_argument('--mr_clip_max', type=float, default=1, help='MR goes from 0 to anywhere at 1800 or only to 1. Here we are setting a relative maximum threshold for MR intensity for better contrast. Value 1 results in no clipping.')
        parser.add_argument('--mr_clip_min', type=float, default=0, help='MR intensities start with 0. Here we are setting a relative minimum value for MR intensity for better contrast. Value 0 results in no clipping.')
        parser.add_argument('--load_size', type=int, default=286, help='scales images after loading [286]. When images are non quadratic see dataset_utils.')
        parser.add_argument('--crop_size', type=int, default=256, help='crops images after scaling (and therefore loading) [256]. When images are non quadratic see dataset_utils.')
        parser.add_argument('--max_dataset_size_train', type=int, default=float('inf'), help='maximum number of samples allowed for training')
        parser.add_argument('--max_dataset_size_val', type=int, default=float('inf'), help='maximum number of samples allowed for validation')
        parser.add_argument('--preprocess', type=str, default='resize_and_crop', help='scaling and cropping images in load time [resize_and_crop | crop | None] ')
        parser.add_argument('--no_flip', type=str, default=True, help='do not flip images unless specified')
        if model == 'CycleGAN':
            parser.add_argument('--lambda_GAN', type=float, default=5, help='weight for cycle loss (A->B->A)')
            parser.add_argument('--lambda_idt', type=float, default=0.5, help='weight of the identity loss in respect to the reconstruction loss. The weight of the identity loss is for example half the weight of the reconstruction loss when lambda_idt = 0.5')
            parser.add_argument('--lambda_grad', type=float, default=0, help='weight of the gradient penalty when synthesising.')
            parser.add_argument('--lambda_D', type=float, default=1, help='weight of the discriminator in his optimization.')
            parser.add_argument('--lambda_MI_A', type=float, default=0, help='weight of the MI-loss between real_A and fake_B')
            parser.add_argument('--lambda_MI_B', type=float, default=0, help='weight of the MI-loss between real_B and fake_A')

        # parameter for validation
        parser.add_argument('--n_hist_bins', type=int, default=100, help='specifies number of bins in clipped value range')
        parser.add_argument('--hist_cut', type=int, default=100, help='value range to be cut on both interval sides for showing main part of histogram')
        self.initialized = True
        return parser

    def gather_options(self):

        if not self.initialized:
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        # get the basic options
        opt, _ = parser.parse_known_args()

        self.parser = parser
        return parser.parse_args()

    def print_options(self, opt):
        """Print and save options

        It prints the options and saves them into a text file / [save_dir | checkpoint_dir] / opt.txt
        """
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

        # save to the disk
        if not opt.phase == 'test':
            expr_dir = os.path.join(opt.save_dir, opt.name)
            util.mkdirs(expr_dir)
            file_name = os.path.join(expr_dir, '{}_opt.txt'.format(opt.phase))
            with open(file_name, 'wt') as opt_file:
                opt_file.write(message)
                opt_file.write('\n')

    def write_csv(self, opt):
        paramsDict = {}
        paramsDict['name'] = opt.name
        paramsDict['n_epochs'] = opt.n_epochs
        paramsDict['n_epochs_decay'] = opt.n_epochs_decay
        paramsDict['batchSize'] = opt.batch_size
        if opt.model == 'CycleGAN':
            paramsDict['lambda_GAN'] = opt.lambda_GAN
            paramsDict['lambda_idt'] = opt.lambda_idt
            paramsDict['lambda_grad'] = opt.lambda_grad
            paramsDict['lambda_D'] = opt.lambda_D
            paramsDict['lambda_MI_A'] = opt.lambda_MI_A
            paramsDict['lambda_MI_B'] = opt.lambda_MI_B
        paramsDict['beta1'] = opt.beta1
        paramsDict['lr'] = opt.lr
        paramsDict['ndf'] = opt.ndf
        paramsDict['ngf'] = opt.ngf
        if opt.model in ['CycleGAN']:
            paramsDict['norm'] = opt.norm
            paramsDict['architectureG'] = opt.netG
            paramsDict['architectureD'] = opt.netD

        paramsDict['init_type'] = opt.init_type
        paramsDict['init_gain'] = opt.init_gain
        paramsDict['CT_clip_max'] = opt.clip
        paramsDict['CT_clip_range'] = opt.clip_range
        paramsDict['MR_clip_max'] = opt.mr_clip_max
        paramsDict['MR_clip_min'] = opt.mr_clip_min
        paramsDict['preprocess'] = opt.preprocess
        paramsDict['no_flip'] = opt.no_flip
        paramsDict['poolSize'] = opt.pool_size
        paramsDict['dataset_size'] = opt.max_dataset_size_train
        paramsDict['val_as_vol'] = opt.val_as_vol
        paramsDict['val_names'] = opt.val_as_vol

        if opt.model == 'CycleGAN':
            fieldnames = ['name', 'n_epochs', 'n_epochs_decay', 'batchSize', 'lambda_GAN', 'lambda_idt', 'lambda_grad', 'lambda_D', 'lambda_MI_A', 'lambda_MI_B', 'beta1',
                          'lr', 'ndf', 'ngf', 'norm', 'architectureG', 'architectureD', 'init_type', 'init_gain', 'CT_clip_max', 'CT_clip_range',
                          'MR_clip_max', 'MR_clip_min', 'preprocess', 'no_flip', 'poolSize', 'dataset_size', 'val_as_vol', 'val_names']

        dictName = opt.platform + '_paramsDict.csv'

        if not os.path.exists(os.path.join(opt.save_dir, dictName)):
            with open(os.path.join(opt.save_dir, dictName), 'a', newline='') as csvfile:
                CSVwriter = csv.DictWriter(csvfile, fieldnames=fieldnames)
                CSVwriter.writeheader()

        with open(os.path.join(opt.save_dir, dictName), 'a', newline='') as csvfile:
            CSVwriter = csv.DictWriter(csvfile, fieldnames=fieldnames)
            CSVwriter.writerow(paramsDict)

    def parse(self):
        opt = self.gather_options()
        opt.isTrain = self.isTrain  # train or test

        self.print_options(opt)

        if self.isTrain:
            self.write_csv(opt)

        # set gpu ids
        str_ids = opt.gpu_ids.split(',')
        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpu_ids.append(id)
        if len(opt.gpu_ids) > 0:
            torch.cuda.set_device(opt.gpu_ids[0])

        self.opt = opt
        return self.opt
