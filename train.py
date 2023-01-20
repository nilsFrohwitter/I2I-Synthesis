from torch.utils.tensorboard import SummaryWriter
from datasets.create_datasets import *
import os
from options.train_options import TrainOptions
from models.init import *
import time
from utils.visualization import *
from utils.util import init_volumes
from utils.writer_funktions import *
from utils.savefile import save_orig_nifti
import torch


opt = TrainOptions().parse()  # get training options

# create summaryWriter for tensorboard
tensorboard_log = os.path.join(opt.save_dir, 'tensorboard_log', opt.name)# str(time.time())) #, datetime.datetime.now().isoformat())

writer = SummaryWriter(log_dir=tensorboard_log)

oneImageTest = False     # True for testing code on one image
if oneImageTest:
    opt.max_dataset_size_train = 1
    opt.max_dataset_size_val = 1
    opt.n_epochs = 10
    opt.n_epochs_decay = 10
    opt.preprocess = 'resize_and_crop'
    opt.save_freq = int((opt.n_epochs + opt.n_epochs_decay)/4)

train_dataset = create_train_dataset(opt)
dataset_size = len(train_dataset)
print('The number of %sing images = %d' % (opt.phase, dataset_size))

if opt.val_as_vol:
    fake_b, fake_a, cycled_b, cycled_a = init_volumes(opt)
    h_f_b, h_c_a, h_f_a, h_c_b = init_volumes(opt)
    hist_volumes = {'h_f_b': h_f_b, 'h_c_a': h_c_a, 'h_f_a': h_f_a, 'h_c_b': h_c_b}
if oneImageTest:
    opt.val_as_vol = False   # can't be together!
    val_dataset = train_dataset
else:
    val_dataset = create_val_dataset(opt)

print('The number of available validation pairs = %d' % len(val_dataset))

model = create_model(opt)
model.set_scheduler(opt)
two_sided_synth = opt.model != 'cs' 

# init some running variables
fixed_data = torch.zeros([2*opt.n_val_images, opt.output_nc, opt.crop_size, opt.crop_size])

iter_count = 0
for epoch in range(opt.n_epochs + opt.n_epochs_decay):
    print('Start Epoch: %d' % (epoch + 1))

    # init timer and running variables
    epoch_start_time = time.time()

    pixel_error = {'AB': [], 'BA': []}
    histograms = {'real_A': torch.zeros([opt.n_hist_bins]), 'real_B': torch.zeros([opt.n_hist_bins]),
                  'fake_B': torch.zeros([opt.n_hist_bins]), 'fake_A': torch.zeros([opt.n_hist_bins]),
                  'cycled_A': torch.zeros([opt.n_hist_bins]), 'cycled_B': torch.zeros([opt.n_hist_bins])}

    for i, data in enumerate(train_dataset):
        iter_count += 1
        model.set_input(data)
        model.optimize_parameters()

        if (i + 1) % 500 == 0:
            print('iter: ' + str(i + 1))

        # get view over loss
        model.update_loss_sum()

    if epoch % opt.save_freq == opt.save_freq - 1:
        model.save_model(epoch + 1)
        do_save_val = True
    else:
        do_save_val = False

    # validation phase
    if opt.val_as_vol:
        for k, val_vol_data in enumerate(val_dataset):  # size:1x256x256x128
            if k % 2 == 0:
                print('val volume: ' + str(k + 1))
            val_size = val_vol_data['A'].shape[-1]
            for sl in range(val_size):
                val_slice = {'A': val_vol_data['A'][:, :, :, sl].unsqueeze(0), 'B': val_vol_data['B'][:, :, :, sl].unsqueeze(0),
                             'A_paths': val_vol_data['A_paths'], 'B_paths': val_vol_data['B_paths']}
                model.set_input(val_slice)
                if not ('resize' in opt.preprocess):    # resized images would be already 256x256 -> no need to split
                    is_splitted = model.split_in_patches()
                else:
                    is_splitted = False
                with torch.no_grad():
                    model.forward()
                    model.calc_loss_D_val()
                    model.calc_loss_G()
                    model.fuse_patches(is_splitted=is_splitted)
                    # save validation images of pairs specified in val_save_pair
                    if do_save_val:
                        fake_b[:, :, sl], cycled_a[:, :, sl], fake_a[:, :, sl], cycled_b[:, :, sl] = model.get_images()
                    if two_sided_synth:
                        pixel_error['AB'].append(model.pixel_error('AB'))
                        pixel_error['BA'].append(model.pixel_error('BA'))
                    else:
                        pixel_error['AB'].append(model.pixel_error('AB'))
                    if do_save_val and k == 0 and sl == 70:  # k=0 because of limited representation capacity, chose slice 70 of 128
                        model.visualize(epoch)   # only visualize a few validations
                        model.visualize_diff(epoch)
                        model.save_writer_diff_images(writer, epoch)
                        model.save_writer_cycled_images(writer, epoch)
                    if k == 0:
                        h_f_b[:, :, sl], h_c_a[:, :, sl], h_f_a[:, :, sl], h_c_b[:, :, sl] = model.get_images()
                        histograms = model.add_histcount(histograms)
            if k == 0:
                write_hist(writer, epoch, opt, val_vol_data['A'], val_vol_data['B'], hist_volumes, val_size)
                write_hist_line(writer, histograms, epoch, opt)
            if k == 0 and epoch == (opt.n_epochs + opt.n_epochs_decay - 1):
                write_final_hist(writer, histograms, opt)
            if do_save_val:
                save_as_nifti(fake_b[:, :, :val_size], cycled_a[:, :, :val_size], opt, val_vol_data['A_paths'][0], epoch, volume=True, name='val',
                              real_dom=opt.data_a, fake_dom=opt.data_b)
                save_as_nifti(fake_a[:, :, :val_size], cycled_b[:, :, :val_size], opt, val_vol_data['B_paths'][0], epoch, volume=True, name='val',
                              real_dom=opt.data_b, fake_dom=opt.data_a)
            if epoch == 0:
                save_orig_nifti(opt, val_vol_data['B_paths'][0], val_vol_data['A_paths'][0], global_reset=True)
    else:
        for k, val_data in enumerate(val_dataset):
            model.set_input(val_data)     # comment out when train on one image pair
            if not ('resize' in opt.preprocess):    # resized images would be already 256x256 -> no need to split
                is_splitted = model.split_in_patches()
            else:
                is_splitted = False
            if k % 100 == 0:
                print('val iter: ' + str(k))
            with torch.no_grad():
                model.forward()
                model.calc_loss_D_val()
                model.calc_loss_G()
                model.fuse_patches(is_splitted=is_splitted)
                # save validation images of pairs specified in val_save_pair
                if do_save_val:
                    if oneImageTest:
                        if opt.data_format == 'nii':
                            model.save_nifti_oneimage('A_paths', epoch)
                            model.save_nifti_oneimage('B_paths', epoch)
                        else:
                            model.save_images(val_data['A_paths'], val_data['B_paths'], epoch)
                    else:
                        if opt.data_format == 'nii':
                            model.save_nifti_images('A_paths', epoch)
                            model.save_nifti_images('B_paths', epoch)
                        else:
                            model.save_images(val_data['A_paths'], val_data['B_paths'], epoch)
                if two_sided_synth:
                    pixel_error['AB'].append(model.pixel_error('AB'))
                    pixel_error['BA'].append(model.pixel_error('BA'))
                else:
                    pixel_error['AB'].append(model.pixel_error('AB'))
                if do_save_val and k == 0:  # k=0 because of limited representation capacity
                    model.visualize(epoch)   # only visualize a few validations
                    model.visualize_diff(epoch)
                    model.save_writer_diff_images(writer, epoch)
                    model.save_writer_cycled_images(writer, epoch)
    if two_sided_synth:
        error_AB_stacked = torch.stack(pixel_error['AB'])
        error_BA_stacked = torch.stack(pixel_error['BA'])
        meanAB = torch.mean(error_AB_stacked, dim=0)
        meanBA = torch.mean(error_BA_stacked, dim=0)
        stdAB = torch.std(error_AB_stacked, dim=0)
        stdBA = torch.std(error_BA_stacked, dim=0)
        print('Epoch %d: pixel error mean | std \n'
              ' is from A->B: %f | %f \n'
              'and from B->A: %f | %f'
              % (epoch, meanAB, stdAB, meanBA, stdBA))
        writer.add_scalar('HU-Diff/meanAB', meanAB, epoch)
        writer.add_scalar('HU-Diff/stdAB', stdAB, epoch)
        writer.add_scalar('HU-Diff/meanBA', meanBA, epoch)
        writer.add_scalar('HU-Diff/stdBA', stdBA, epoch)
    else:
        error_AB_stacked = torch.stack(pixel_error['AB'])
        meanAB = torch.mean(error_AB_stacked, dim=0)
        stdAB = torch.std(error_AB_stacked, dim=0)
        print('Epoch %d: pixel error mean | std \n'
              ' is from A->B: %f | %f'
              % (epoch, meanAB, stdAB))
        writer.add_scalar('HU-Diff/meanAB', meanAB, epoch)
        writer.add_scalar('HU-Diff/stdAB', stdAB, epoch)

    model.update_learning_rate()

    # saving avg losses of current epoch in tensorboard writer
    model.add_losses_to_writer(writer, dataset_size, epoch)

    print('End of epoch %d / %d \t Time Taken: %d sec \t images iterated: %d' %
          (epoch + 1, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time, iter_count * opt.batch_size))
