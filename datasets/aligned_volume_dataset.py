"""
This code is highly related to the PyTorch-CycleGAN implementaion
from Zhu et al.:
https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix

"""

from torch.utils.data import Dataset
from .dataset_utils import *


class AlignedVolumeDataset(Dataset):
    """This class is for aligned volume datasets (would be for validation or
    testing of given data volumes)

    """

    def __init__(self, opt, d_type, mode):
        if opt.phase == 'train':
            self.root_A = opt.dataroot_val_volume
            self.root_B = opt.dataroot_val_volume
            self.mode = 'val'
        elif opt.phase == 'test':
            if opt.dataroot_a:
                self.root_A = os.path.join(opt.dataroot_a, 'test', d_type)
            else:
                self.root_A = os.path.join(opt.dataroot, 'test', d_type)
            self.root_B = os.path.join(opt.dataroot, 'test', d_type)
            self.mode = 'test'
        else:
            print('AlignedVolumeDataset should only be for validation or testing')

        self.opt = opt

        if opt.do_LOOCV:
            self.A_paths = sorted(make_dataset(opt, self.root_A, max_dataset_size=opt.max_dataset_size_val, right_domain=opt.data_a, mode=mode))
            self.B_paths = sorted(make_dataset(opt, self.root_B, max_dataset_size=opt.max_dataset_size_val, right_domain=opt.data_b, mode=mode))
        else:
            self.A_paths = sorted(make_dataset(opt, self.root_A, max_dataset_size=opt.max_dataset_size_val, right_domain=opt.data_a))
            self.B_paths = sorted(make_dataset(opt, self.root_B, max_dataset_size=opt.max_dataset_size_val, right_domain=opt.data_b))
        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)
        assert(self.A_size == self.B_size)  # A and B should contain the same number of volumes
        self.transform = get_volume_transform(opt, grayscale=(opt.input_nc == 1), mode=self.mode)
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
        A_vol = load_nifti_vol(A_path)
        B_vol = load_nifti_vol(B_path)
        # apply image transformation
        A = center_crop_vol(A_vol, self.opt)
        B = center_crop_vol(B_vol, self.opt)
        A = clip_and_scale(A, self.opt, self.opt.data_a)
        B = clip_and_scale(B, self.opt, self.opt.data_b)
        A_sized = torch.zeros([min(self.opt.crop_size, self.opt.img_size[0]),
                               min(self.opt.crop_size, self.opt.img_size[1]),
                               A.size()[-1]])
        B_sized = A_sized.clone()
        for i in range(A.size()[-1]):
            A_sized[:, :, i] = self.transform(A[:, :, i])
            B_sized[:, :, i] = self.transform(B[:, :, i])

        return {'A': A_sized, 'B': B_sized, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        """return the number of images in the dataset.
        For a different number of images, we take the maximum
        """
        return max(self.A_size, self.B_size)


def get_volume_transform(opt, grayscale=False, convert=True, mode=None):
    """Returns the composed transform list.
    For validation or testing mode must be defined
    (default is None, for training)."""
    transform_list = []
    transform_list.append(transforms.ToPILImage())
    if mode != 'test':
        if 'resize' in opt.preprocess and (opt.crop_size < opt.img_size[0] or opt.crop_size < opt.img_size[1]):
            transform_list.append(transforms.Resize([opt.crop_size, opt.crop_size], interpolation=Image.BICUBIC))
    if convert:
        transform_list += [transforms.ToTensor()]
        if grayscale:
            transform_list += [transforms.Normalize((0.5,), (0.5,))]
        else:
            transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)


def center_crop_vol(vol, opt):
    """This method crops the input volume to the region of interest.
    E. g. because of bench artefacts in CT imaging, the middle of the anatomic
    region is extracted befor training.
    """
    vol_shape = vol.shape
    # print('shape of cropped img is: ' + str(img_shape))
    h = opt.center_crop_height
    w = opt.center_crop_width
    if vol_shape[0] <= h or vol_shape[1] <= w:
        return vol
    else:
        return vol[int((vol_shape[0] - h) / 2):int((vol_shape[0] + h) / 2),
                   int((vol_shape[1] - w) / 2):int((vol_shape[1] + w) / 2), :]


def load_nifti_vol(path):
    vol = nibabel.load(path)
    # worldMatrix = vol.affine  # .header.get_zooms()
    vol = vol.get_data().astype(float).transpose(1, 0, 2)
    vol = torch.from_numpy(vol).float()
    return vol  # , worldMatrix, vol.header
