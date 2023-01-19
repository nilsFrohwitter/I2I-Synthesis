"""
This code is highly related to the PyTorch-CycleGAN implementaion
from Zhu et al.:
https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix

"""

from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    """This class includes the training options to the already definded BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)

        parser.add_argument('--phase', type=str, default='train', help='train | val | test')
        parser.add_argument('--n_epochs', type=int, default=100, help='number of epochs with fixed lr')
        parser.add_argument('--n_epochs_decay', type=int, default=100, help='number of epochs with linear decay of lr to zero')
        parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam optimizer')
        parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate')
        parser.add_argument('--pool_size', type=int, default=100, help='size of image pools storing generated images')
        parser.add_argument('--n_val_images', type=int, default=1, help='number of images from the validation dataset to monitor the training')

        self.isTrain = True
        return parser
