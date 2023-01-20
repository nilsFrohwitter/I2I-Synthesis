from datasets.create_datasets import create_test_dataset
from options.test_options import TestOptions
from models.init import *
from utils.util import init_volumes
from utils.savefile import save_as_nifti, save_orig_nifti
from evaluation.hist_plot import *
import torch
import os


opt = TestOptions().parse()  # get training options
test_dataset = create_test_dataset(opt)
dataset_size = len(test_dataset)
print('The number of %sing images = %d' % (opt.phase, dataset_size))

# load checkpoint
checkpoint_name = '22-02-23_18-06-01_e0c6ebba-94ca-11ec-88a1-ac1f6b89b030'
state = 'checkpoints/epoch_50_02-24_09-25-19.pth'
opt.checkpoint_dir = os.path.join(opt.save_dir, checkpoint_name, state)
opt.name = checkpoint_name
left_hist_cut_only = False

model = create_model(opt)
model.load_model()
model.eval()

iter_count = 0
pixel_error = {'AB': [], 'BA': []}
do_save_nifti = True
fake_b, fake_a, cycled_b, cycled_a = init_volumes(opt)
h_real_a, h_real_b, h_fake_b, h_cycled_a, h_fake_a, h_cycled_b = init_hists(opt, dataset_size)


already_tested = False
if opt.test_as_vol:
    for k, test_vol_data in enumerate(test_dataset):  #  for ex.: size:1x256x256x128
        num_slices = test_vol_data['A'].shape[-1]
        for sl in range(num_slices):
            test_slice = {'A': test_vol_data['A'][:, :, :, sl].unsqueeze(0),
                         'B': test_vol_data['B'][:, :, :, sl].unsqueeze(0),
                         'A_paths': test_vol_data['A_paths'], 'B_paths': test_vol_data['B_paths']}
            model.set_input(test_slice)
            if not ('resize' in opt.preprocess):  # resized images would be already 256x256 -> no need to split
                is_splitted = model.split_in_patches()
            else:
                is_splitted = False
            with torch.no_grad():
                model.forward()
                model.fuse_patches(is_splitted=is_splitted)
                # save all test volumes (real, fake and cycled) but first: collect fake/cycled slices
                if do_save_nifti:
                    fake_b[:, :, sl], cycled_a[:, :, sl], fake_a[:, :, sl], cycled_b[:, :, sl] = model.get_images()
                pixel_error['AB'].append(model.pixel_error('AB'))
                pixel_error['BA'].append(model.pixel_error('BA'))
                # if k % 10 == 0 and sl == 70:  # only visualize a few test volumes; chose slice 70 of 128
                #     model.visualize()
                #     model.visualize_diff()
        if do_save_nifti and not already_tested:
            save_as_nifti(fake_b[:, :, :num_slices], cycled_a[:, :, :num_slices], opt, test_vol_data['A_paths'][0], volume=True, real_dom=opt.data_a,
                          fake_dom=opt.data_b)
            save_as_nifti(fake_a[:, :, :num_slices], cycled_b[:, :, :num_slices], opt, test_vol_data['B_paths'][0], volume=True, real_dom=opt.data_b,
                          fake_dom=opt.data_a)
            save_orig_nifti(opt, data_b_dir=test_vol_data['B_paths'][0], data_a_dir=test_vol_data['A_paths'][0], global_reset=True)

        h_real_a[k, :], h_real_b[k, :], h_fake_b[k, :], h_cycled_a[k, :], h_fake_a[k, :], h_cycled_b[k, :] \
            = matplot_hist(test_vol_data['A'].squeeze(0), test_vol_data['B'].squeeze(0),
                           fake_b[:, :, :num_slices], cycled_a[:, :, :num_slices], 
                           fake_a[:, :, :num_slices], cycled_b[:, :, :num_slices], 
                           opt, vol=k, do_plot=False, left_cut_only=left_hist_cut_only)
    matplot_hist_avg(h_real_a, h_real_b, h_fake_b, h_cycled_a, h_fake_a, h_cycled_b, opt, left_cut_only=left_hist_cut_only)
    error_AB_stacked = torch.stack(pixel_error['AB'])
    error_BA_stacked = torch.stack(pixel_error['BA'])
    meanAB = torch.mean(error_AB_stacked, dim=0)
    meanBA = torch.mean(error_BA_stacked, dim=0)
    stdAB = torch.std(error_AB_stacked, dim=0)
    stdBA = torch.std(error_BA_stacked, dim=0)
    print('pixel error mean | std \n'
          ' is from A->B: %f | %f \n'
          'and from B->A: %f | %f'
          % (meanAB, stdAB, meanBA, stdBA))

elif not already_tested:
    for k, test_data in enumerate(test_dataset):
        model.set_input(test_data)
        is_splitted = model.split_in_patches()
        if k % 401 == 400:
            print('test iter: ' + str(k))
        with torch.no_grad():
            model.forward()
            model.fuse_patches(is_splitted=is_splitted)
            pixel_error['AB'].append(model.pixel_error('AB'))
            pixel_error['BA'].append(model.pixel_error('BA'))
            if k % (round(dataset_size/8)) == 0:  # k=0 because of limited representation capacity
                model.visualize()   # only visualize a few validations
                model.visualize_diff()
    error_AB_stacked = torch.stack(pixel_error['AB'])
    error_BA_stacked = torch.stack(pixel_error['BA'])
