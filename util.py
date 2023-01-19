import os
import torch


def mkdirs(paths):
    """create empty directories if they don't exist

    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """create a single empty directory if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)


def check_val_dirs(opt, name, epoch):
    save_dir_images = os.path.join(opt.save_dir, opt.name, 'val_images', 'epoch_' + format(epoch, '02d'))
    mkdir(save_dir_images)
    for nm in name:
        fake_a_dir = os.path.join(save_dir_images, nm + '_fake_' + opt.data_a)
        cylced_b_dir = os.path.join(save_dir_images, nm + '_cycled_' + opt.data_b)
        fake_b_dir = os.path.join(save_dir_images, nm + '_fake_' + opt.data_b)
        cylced_a_dir = os.path.join(save_dir_images, nm + '_cycled_' + opt.data_a)
        mkdirs([fake_a_dir, fake_b_dir, cylced_a_dir, cylced_b_dir])


def check_test_dirs(opt):
    save_dir_images = os.path.join(opt.save_dir, opt.name, 'test_images')
    already_tested = os.path.exists(save_dir_images)
    if not already_tested:
        mkdir(save_dir_images)
        real_a_dir = os.path.join(save_dir_images, 'real_' + opt.data_b)
        fake_a_dir = os.path.join(save_dir_images, 'fake_' + opt.data_b)
        cycled_a_dir = os.path.join(save_dir_images, 'cycled_' + opt.data_b)
        real_b_dir = os.path.join(save_dir_images, 'real_' + opt.data_a)
        fake_b_dir = os.path.join(save_dir_images, 'fake_' + opt.data_a)
        cylced_b_dir = os.path.join(save_dir_images, 'cycled_' + opt.data_a)
        mkdirs([real_a_dir, fake_a_dir, cycled_a_dir, real_b_dir, fake_b_dir, cylced_b_dir])
    return already_tested


def init_volumes(opt):
    fake_ct = torch.zeros([min(opt.img_size[0], opt.crop_size), min(opt.img_size[1], opt.crop_size), 192])
    cycled_cbct = fake_ct.clone()
    fake_cbct = fake_ct.clone()
    cycled_ct = fake_ct.clone()
    return fake_ct, cycled_cbct, fake_cbct, cycled_ct
