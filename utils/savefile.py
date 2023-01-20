import os
import torch
import datetime
import shutil
from torchvision.utils import save_image
import nibabel as nib
import numpy as np
import torchvision.transforms as transforms
from PIL import Image


def save(models, optimizers, save_dir, epoch, losses):
    timestamp = datetime.datetime.now().strftime('_%m-%d_%H-%M-%S')
    torch.save({
        'epoch': epoch,
        'G_A_state_dict': models['G_A'].state_dict(),
        'G_B_state_dict': models['G_B'].state_dict(),
        'D_A_state_dict': models['D_A'].state_dict(),
        'D_B_state_dict': models['D_B'].state_dict(),
        'optim_G_state_dict': optimizers['optim_G'].state_dict(),
        'optim_D_state_dict': optimizers['optim_D'].state_dict(),
        'loss_G': losses['loss_G'],
        'loss_D_A': losses['loss_D_A'],
        'loss_D_B': losses['loss_D_B'],
    }, os.path.join(save_dir, 'epoch_' + format(epoch, '02d') + timestamp + '.pth'))


def save_state(models, optimizers, losses, epoch, opt):
    save_state_dir = os.path.join(opt.save_dir, opt.name, 'checkpoints')

    if not os.path.exists(save_state_dir):
        os.makedirs(save_state_dir)

    save(models, optimizers, save_state_dir, epoch, losses)
    return


def save_best_states(models, optimizers, save_dir, epoch, losses):
    save_best_states_dir = os.path.join(save_dir, 'best_state')

    try:
        os.mkdir(save_best_states_dir)
    except FileExistsError:
        shutil.rmtree(save_best_states_dir)
        os.mkdir(save_best_states_dir)

    save(models, optimizers, save_best_states_dir, epoch, losses)
    return


def load_state(models, optimizers, load_dir):
    checkpoint = torch.load(load_dir)
    models['G_A'].load_state_dict(checkpoint['G_A_state_dict'])
    models['G_B'].load_state_dict(checkpoint['G_B_state_dict'])
    models['D_A'].load_state_dict(checkpoint['D_A_state_dict'])
    models['D_B'].load_state_dict(checkpoint['D_B_state_dict'])
    optimizers['optim_G'].load_state_dict(checkpoint['optim_G_state_dict'])
    optimizers['optim_D'].load_state_dict(checkpoint['optim_D_state_dict'])
    epoch = checkpoint['epoch']
    return models, optimizers, epoch


def load_state_test(models, load_dir):
    checkpoint = torch.load(load_dir)
    models['G_A'].load_state_dict(checkpoint['G_A_state_dict'])
    models['G_B'].load_state_dict(checkpoint['G_B_state_dict'])
    # epoch = checkpoint['epoch']
    return models


def save_images(img_a, img_b, a_path, b_path, epoch, state, type_, opt):
    """saves the images in the save_dir under the same name as from the data

    Already saved images of the used checkpoint will be overwritten with latest test run.
    """

    # need to extract the image name (.../<name.jpg>)
    if opt.platform == 'Windows':
        path_a_split = a_path[0].rsplit(sep='\\', maxsplit=1)
        path_b_split = b_path[0].rsplit(sep='\\', maxsplit=1)
    else:
        path_a_split = a_path[0].rsplit(sep='/', maxsplit=1)
        path_b_split = b_path[0].rsplit(sep='/', maxsplit=1)
    # "unnormalize" images
    img_b = img_b / 2 + 0.5
    img_a = img_a / 2 + 0.5

    # create folders for generated images
    a_save_dir = os.path.join(opt.save_dir, opt.name, state, 'epoch_' + str(epoch+1), type_ + 'Horses')  # fakeImagesB
    b_save_dir = os.path.join(opt.save_dir, opt.name, state, 'epoch_' + str(epoch+1), type_ + 'Zebras')  # fakeImagesA

    if not os.path.exists(a_save_dir):
        os.makedirs(a_save_dir)
        os.mkdir(b_save_dir)

    # save the generated images
    if type_ == 'fake': 
        b_dir = os.path.join(b_save_dir, path_a_split[-1])
        a_dir = os.path.join(a_save_dir, path_b_split[-1])
    else:
        b_dir = os.path.join(b_save_dir, path_b_split[-1])
        a_dir = os.path.join(a_save_dir, path_a_split[-1])

    if os.path.exists(b_dir):
        os.remove(b_dir)
        save_image(img_b.squeeze(0), b_dir)  # squeeze to remove batch dimension
    else:
        save_image(img_b.squeeze(0), b_dir)

    if os.path.exists(a_dir):
        os.remove(a_dir)
        save_image(img_a.squeeze(0), a_dir)
    else:
        save_image(img_a.squeeze(0), a_dir)


def save_as_nifti(fake, cycled, opt, img_dir, epoch=None, volume=False, name=None, real_dom=None, fake_dom=None):
    if opt.platform == 'Windows':
        sep = '\\'
    else:
        sep = '/'

    if opt.phase == 'train':
        save_dir_images = os.path.join(opt.save_dir, opt.name, 'val_images', 'epoch_' + format(epoch + 1, '02d'))
        fake_dir = os.path.join(save_dir_images, name + '_fake_' + fake_dom)
        cycled_dir = os.path.join(save_dir_images, name + '_cycled_' + real_dom)
    elif opt.phase == 'test':
        save_dir_images = os.path.join(opt.save_dir, opt.name, 'test_images')
        fake_dir = os.path.join(save_dir_images, 'fake_' + fake_dom)
        cycled_dir = os.path.join(save_dir_images, 'cycled_' + real_dom)
        name = img_dir.rsplit(sep)[-3]
    else:
        print('something went wrong; opt.phase should be [train|test]')

    fake = fake / 2 + 0.5
    cycled = cycled / 2 + 0.5

    fake_resized = torch.zeros([opt.img_size[0], opt.img_size[1],fake.size()[-1]])
    cycled_resized = fake_resized.clone().detach()

    # <transform> is specific for 512x512 CBCT-CT nifti images
    transform = transforms.Compose([transforms.ToPILImage(),
                                    transforms.Resize([opt.center_crop_width, opt.center_crop_height], interpolation=Image.BICUBIC),
                                    transforms.Pad(padding=int((512-opt.center_crop_height)/2), fill=0, padding_mode='constant'),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,))])
    transform_MRCT = transforms.Compose([transforms.ToPILImage(),
                                    transforms.Resize([opt.img_size[0], opt.img_size[1]], interpolation=Image.BICUBIC),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,))])

    if volume:
        num_slices = fake.size()[-1]
        for i in range(num_slices):
            fake_resized[:, :, i] = transform_MRCT(fake[:, :, i])
            cycled_resized[:, :, i] = transform_MRCT(cycled[:, :, i])

        fake_name = 'fake_' + fake_dom + '_' + name + '_' + img_dir.rsplit(sep, 1)[-1]
        cycled_name = 'cycled_' + real_dom + '_' + name + '_' + img_dir.rsplit(sep, 1)[-1]

        fake_resized = normalized_to_original_values(fake_resized, opt, domain = fake_dom)
        cycled_resized = normalized_to_original_values(cycled_resized, opt, domain = real_dom)
        fake_nii = nib.Nifti1Image(np.transpose(fake_resized[:, :, :num_slices].numpy(), (1, 0, 2)), affine=None)
        cycled_nii = nib.Nifti1Image(np.transpose(cycled_resized[:, :, :num_slices].numpy(), (1, 0, 2)), affine=None)
    else:
        fake_name = 'fake_' + fake_dom + '_' + img_dir.rsplit(sep, 1)[-1]
        cycled_name = 'cycled_' + real_dom + '_' + img_dir.rsplit(sep, 1)[-1]
        fake = normalized_to_original_values(transform_MRCT(fake.squeeze(0)), opt, domain = fake_dom)
        cycled = normalized_to_original_values(transform_MRCT(cycled.squeeze(0)), opt, domain = real_dom)
        fake_nii = nib.Nifti1Image(np.transpose(fake.squeeze(0).cpu().numpy(), (1, 0)), affine=None)
        cycled_nii = nib.Nifti1Image(np.transpose(cycled.squeeze(0).cpu().numpy(), (1, 0)), affine=None)
    if not os.path.exists(fake_dir):
        os.makedirs(fake_dir)
        os.makedirs(cycled_dir)
    nib.save(fake_nii, os.path.join(fake_dir, fake_name))
    nib.save(cycled_nii, os.path.join(cycled_dir, cycled_name))


def save_orig_nifti(opt, data_b_dir, data_a_dir, global_reset=False):
    if opt.platform == 'Windows':
        sep = '\\'
    else:
        sep = '/'

    if opt.phase == 'test':
        save_dir_images = os.path.join(opt.save_dir, opt.name, 'test_images')
    else:
        save_dir_images = os.path.join(opt.save_dir, opt.name, 'val_images')
    real_b_dir = os.path.join(save_dir_images, 'real_' + opt.data_b)
    real_a_dir = os.path.join(save_dir_images, 'real_' + opt.data_a)

    if not os.path.exists(real_a_dir):
        os.makedirs(real_a_dir)
        os.makedirs(real_b_dir)

    data_b_name = 'real_' + opt.data_b + '_' + data_b_dir.rsplit(sep, 1)[-1]
    data_a_name = 'real_' + opt.data_a + '_' + data_a_dir.rsplit(sep, 1)[-1]

    data_b_nii = nib.load(data_b_dir)
    data_a_nii = nib.load(data_a_dir)

    b_data = torch.from_numpy(data_b_nii.get_fdata())
    a_data = torch.from_numpy(data_a_nii.get_fdata())

    b_clip = b_data.clone().detach()
    a_clip = a_data.clone().detach()

    if opt.data_a == 'MR':
        a_data_max = a_data.max()
        for i in range(b_clip.size()[-1]):
            b_clip[:, :, i] = torch.clamp(b_data[:, :, i], min=opt.clip - opt.clip_range, max=opt.clip)
            a_clip[:, :, i] = torch.clamp(a_data[:, :, i], min=a_data_max * opt.mr_clip_min, max=a_data_max * opt.mr_clip_max)
    else:
        for i in range(b_clip.size()[-1]):
            b_clip[:, :, i] = torch.clamp(b_data[:, :, i], min=opt.clip - opt.clip_range, max=opt.clip)
            a_clip[:, :, i] = torch.clamp(a_data[:, :, i], min=opt.clip - opt.clip_range, max=opt.clip)

    if global_reset:
        b_clip_nii = nib.Nifti1Image(b_clip.numpy(), affine=None)
        a_clip_nii = nib.Nifti1Image(a_clip.numpy(), affine=None)
    else:
        b_clip_nii = nib.Nifti1Image(b_clip, data_b_nii.affine, data_b_nii.header)
        a_clip_nii = nib.Nifti1Image(a_clip, data_a_nii.affine, data_a_nii.header)

    nib.save(b_clip_nii, os.path.join(real_b_dir, data_b_name))
    nib.save(a_clip_nii, os.path.join(real_a_dir, data_a_name))


def normalized_to_original_values(data, opt, domain):
    if domain == 'mr':
        return (data/2 + 0.5)  # * opt.mr_clip_max
    else:
        return (data/2 + 0.5) * opt.clip_range + (opt.clip - opt.clip_range)
