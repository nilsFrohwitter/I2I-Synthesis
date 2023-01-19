"""
This code is highly related to the PyTorch-CycleGAN implementaion
from Zhu et al.:
https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix

"""

from .base_options import BaseOptions


class TestOptions(BaseOptions):
    """This class includes test options to the already defined BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('--phase', type=str, default='test', help='[train, val, test]')
        parser.add_argument('--eval', action='store_true', help='eval mode during test time')
        # avoid cropping when testing, so load_size should be the same size as crop_size
        parser.set_defaults(load_size=parser.get_default('crop_size'))
        parser.add_argument('--n_test_images', type=int, default=float('Inf'), help='number of images for testing per patient alignment')
        parser.add_argument('--test_as_vol', type=str, default=True, help='set on True if test data is available as volumes, otherwise False')

        self.isTrain = False
        return parser