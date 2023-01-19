"""
This code is highly related to the PyTorch-CycleGAN implementaion
from Zhu et al.:
https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix

"""

from torch.utils.data import Dataset
from .dataset_utils import *


class AlignedDataset(Dataset):
    """This class is for aligned datasets.

    """

    def __init__(self, opt, d_type, mode):
        self.root = opt.dataroot
        self.opt = opt

        if opt.phase == 'train':
            self.mode = 'val'
        else:
            self.mode = 'test'
        if opt.data_a == 'pferd':
            if opt.dataroot_a:
                self.dir_A = os.path.join(opt.dataroot_a, self.mode)
            else:
                self.dir_A = os.path.join(self.root, self.mode)
            self.dir_B = os.path.join(self.root, self.mode)
        else:
            if opt.dataroot_a:
                self.dir_A = os.path.join(opt.dataroot_a, self.mode, d_type)
            else:
                self.dir_A = os.path.join(self.root, self.mode, d_type)
            self.dir_B = os.path.join(self.root, self.mode, d_type)

        self.A_paths = sorted(make_dataset(opt, self.dir_A, max_dataset_size=opt.max_dataset_size_val, right_domain=opt.data_a))
        self.B_paths = sorted(make_dataset(opt, self.dir_B, max_dataset_size=opt.max_dataset_size_val, right_domain=opt.data_b))
        self.A_paths, self.B_paths = shuffle_list(self.A_paths, self.B_paths, opt=opt)
        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)
        assert(self.A_size == self.B_size)  # A and B should contain the same number of images
        self.transform = get_transform(opt, grayscale=(opt.input_nc == 1), mode=self.mode)
        self.input_nc = opt.input_nc
        self.output_nc = opt.output_nc

    def __getitem__(self, index):
        """Return a data point and its metadata information.
                Parameters:
                    index (int)      -- a random integer for data indexing
                Returns a dictionary that contains A, B, A_paths and B_paths
                    A (tensor)       -- an image in the input domain
                    B (tensor)       -- its corresponding image in the target domain
                    A_paths (str)    -- image paths
                    B_paths (str)    -- image paths
                """
        A_path = self.A_paths[index % self.A_size]  # make sure index is within range
        B_path = self.B_paths[index % self.B_size]
        if self.input_nc == 3:
            A = Image.open(A_path).convert('RGB')   # convert to RGB
            B = Image.open(B_path).convert('RGB')
        elif self.input_nc == 1:
            A_img = load_nifti(A_path)
            B_img = load_nifti(B_path)
        # apply image transformation
            A = center_crop(A_img, self.opt)
            B = center_crop(B_img, self.opt)
            A = clip_and_scale(A, self.opt, self.opt.data_a)
            B = clip_and_scale(B, self.opt, self.opt.data_b)
        A = self.transform(A)
        B = self.transform(B)

        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        """return the number of images in the dataset.
        For a different number of images, we take the maximum
        """
        return max(self.A_size, self.B_size)
