"""
This code is highly related to the PyTorch-CycleGAN implementaion
from Zhu et al.:
https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix

"""

import random
import torchvision.transforms as transforms
import os
from PIL import Image
import nibabel
import torch
import numpy as np


def center_crop(img, opt):
    """This method crops the input image to the region of interest.
    Because of bench artefacts in CT imaging, the middle part of the anatomic 
    region is extracted befor training.
     """
    img_shape = img.shape
    h = opt.center_crop_height
    w = opt.center_crop_width
    if img_shape[0] <= h or img_shape[1] <= w:
        return img  # return img unchanged if croped dimensions are bigger 
    else:
        return img[int((img_shape[0] - h) / 2):int((img_shape[0] + h) / 2),
                   int((img_shape[1] - w) / 2):int((img_shape[1] + w) / 2)]


def get_transform(opt, grayscale=False, convert=True, mode=None):
    """returns the composed transform list.
    For validation or testing <mode> must be defined (default is None, for training)."""
    if opt.img_size[0] == opt.img_size[1]:
        load_a = opt.load_size
        load_b = opt.load_size
        crop_a = opt.crop_size
        crop_b = opt.crop_size
    else:
        load_a = np.flat(opt.img_size[0] * 1.09375)  # e.g. 160->175
        load_b = np.flat(opt.img_size[1] * 1.09375)  # e.g. 192->210
        crop_a = opt.img_size[0]
        crop_b = opt.img_size[1]

    transform_list = []
    if opt.data_format == 'nii':
        transform_list.append(transforms.ToPILImage())
    if 'resize' in opt.preprocess:
        transform_list.append(transforms.Resize([load_a, load_b], interpolation=Image.BICUBIC))
    if 'crop' in opt.preprocess:
        if mode:
            transform_list.append(transforms.CenterCrop(size=[crop_a, crop_b]))
        else:
            transform_list.append(transforms.RandomCrop(size=[crop_a, crop_b]))
    if not (opt.no_flip or mode is not None):
        transform_list.append(transforms.RandomHorizontalFlip())
    if convert:
        transform_list += [transforms.ToTensor()]
        # Normalize --> range[-1, 1]
        if grayscale:
            transform_list += [transforms.Normalize((0.5,), (0.5,))]
        else:
            transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)


def shuffle_list(*ls, opt):
    _l = list(zip(*ls))
    random.shuffle(_l)
    return zip(*_l)


IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG', '.nii', '.gz', '.png'
]


def is_image_file(filename):
    if filename.endswith('labels.nii'):
        return False
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(opt, directory, max_dataset_size=float("inf"), right_domain=None, mode=None):
    elements = []
    assert os.path.isdir(directory), '%s is not a valid directory' % directory

    for root, _, fnames in sorted(os.walk(directory)):
        for fname in fnames:
            if not opt.do_LOOCV:
                if is_image_file(fname) and (not right_domain or right_domain in fname):
                    path = os.path.join(root, fname)
                    elements.append(path)
            elif any(vol + '_tcia' in fname for vol in opt.val_names) and mode=='train':
                continue
            elif not any(vol + '_tcia' in fname for vol in opt.val_names) and mode=='val':
                continue
            else:
                if is_image_file(fname) and (not right_domain or right_domain in fname):
                    path = os.path.join(root, fname)
                    elements.append(path)
    if mode != 'train':
        elements.sort()
    return elements[:min(max_dataset_size, len(elements))]


def load_nifti(path):
    img = nibabel.load(path)
    # worldMatrix = img.affine  # .header.get_zooms()
    img = img.get_data().astype(float).transpose(1, 0)
    img = torch.from_numpy(img).float()
    return img


def transform_to_HU(img, opt):
    """transforms from image range [-1,1] to [0,opt.clip], where opt.clip is
    the upper threshold for the loaded image intensities
    """
    img = img / 2 + 0.5
    img = img * opt.clip_range + (opt.clip - opt.clip_range)
    return img


def clip_and_scale(img, opt, d_type=None):
    """clips the intensity values of the image to a defined range (given by 
    parameters in opt) and scales the image to the range [0, 1].
    This is only used for MR or CT images.
    """
    if d_type == 'MR':
        img = torch.clamp(img, min=img.max() * opt.mr_clip_min, max=img.max() * opt.mr_clip_max)
    else:
        img = torch.clamp(img, min=opt.clip - opt.clip_range, max=opt.clip)

    # scaling all values to [0,1]
    img = img - img.min()
    img = img / img.max()
    return img
