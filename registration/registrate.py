import os
import pickle
import datetime
from .registration_tools import *
from evaluation.eval_reg import calculate_initial_dice_values, calculate_dice_values

"""
This registration is used on MR-CT image data.
Parameter:
    pat : patient number; this is the suffix in their file name
          "None" means it is not a LOOCV --> all data in /test_images is registrated
    run : specifies the name of the saved model
    
For LOOCV, every run contains a model which has been trained on all training
(and validation) data except the corresponding <pat>.  
"""
pat = None
# run = '22-05-09_14-13-18_696fbb76-cf91-11ec-beba-ac1f6bf5ab70'

# now the LOOCV: in Order 02, 04, 06, 08, 10, 12, 14, 16
# pat = '02'
# run = '22-05-11_10-46-49_e5f90ec4-d106-11ec-ab45-ac1f6bf5ab70'
# pat = '04'
# run = '22-05-10_17-24-34_4bc31b56-d075-11ec-a90a-ac1f6bf8ff9c'
# pat = '06'
# run = '22-05-10_17-37-31_1afa2fda-d077-11ec-8cbf-ac1f6bf5ab70'
# pat = '08'
# run = '22-05-10_17-52-01_21a855d0-d079-11ec-8a52-ac1f6bf5ab70'
# pat = '10'
# run = '22-05-11_08-39-01_0af31880-d0f5-11ec-bd41-ac1f6bf8ff9c'
# pat = '12'
# run = '22-05-11_10-50-33_6b68bb18-d107-11ec-8c8e-ac1f6bf8ff9c'
# pat = '14'
# run = '22-05-11_10-55-42_233c8e04-d108-11ec-b93a-ac1f6bf8ff9c'
pat = '16'
run = '22-05-11_13-15-14_a183f906-d11b-11ec-a9a9-ac1f6bf8ff9c'

methods = ['Rigid', 'SyN', 'SyNRA', 'ElasticSyN']
image_origin = "val_images"  # "test_images" or "val_images"
do_val = image_origin == "val_images"      # if reg is to be validated
framework = 'ANTs'   # ['elastix', 'ANTs', 'corrfield'], only ANTs works for now
overwrite_if_exists = False

date = datetime.datetime.now().strftime('%y-%m-%d_%H-%M-%S')

data_root = '/media/sf_sharedfoulderVM/val_images/' + run
segm_dir = "/media/sf_sharedfoulderVM/val_images/all_seg"
dice_save_dir = "/media/sf_sharedfoulderVM/results"
sep = os.path.sep


if framework != 'ANTs':
    methods = [None]

for _, method in enumerate(methods):
    print('starting with ' + str(method))

    save_dir = '/media/sf_sharedfoulderVM/results/' + run + '/reg_results_new2'

    if framework == 'ANTs':
        save_dir = save_dir + '_' + method
    else:
        save_dir = save_dir + '_' + framework

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    dice_scores = {'init': [], 'rMRrCT': [], 'fCTrCT': [], 'rMRfMR': []}
    dice_init = []
    dice_rMRrCT = []
    dice_fCTrCT = []
    dice_rMRfMR = []

    # running for registration real <-> real
    temp_dirs_1, ref_dirs_1 = get_dirs(data_root, 'all_real_MR', 'all_real_CT', pat=pat)
    if do_val:
        seg_temp_dirs_1, seg_ref_dirs_1 = get_dirs(segm_dir, 'all_real_MR', 'all_real_CT', pat=pat)

    if framework != 'ANTs':
        case = 'case0'
    else:
        case = None

    for index, item in enumerate(temp_dirs_1):
        print('index is: %d', index)
        name_1 = item.rsplit(sep, maxsplit=1)[-1]
        name_1 = cut_extention(name_1)
        name_2 = ref_dirs_1[index].rsplit(sep, maxsplit=1)[-1]
        moved_1, def_1, fixed_1, moved_grid_1 = registrate_vols(index, item, ref_dirs_1[index], save_dir, method, framework, reg_case=case)

        if do_val:
            if framework == 'corrfield':
                fixed = fixed_1
            dice_scores['init'].append(calculate_initial_dice_values(seg_temp_dirs_1[index], seg_ref_dirs_1[index]))
            dice_scores['rMRrCT'].append(calculate_dice_values(seg_temp_dirs_1[index], seg_ref_dirs_1[index], def_1, framework, save_dir=save_dir, case=case))
            # raise NameError('debugging')
            dice_init.append(gather_dice(seg_temp_dirs_1[index], seg_ref_dirs_1[index], run, method, image_origin, framework, save_dir, case=case))
            dice_rMRrCT.append(gather_dice(seg_temp_dirs_1[index], seg_ref_dirs_1[index], run, method, image_origin, framework, save_dir, def_1, case=case))
        image_write(moved_1, save_dir + '/' + "rMR-rCT_moved_" + name_1 + name_2, framework)
        image_write(fixed_1, save_dir + '/' + "rMR-rCT_fixed_" + name_1 + name_2, framework)
        image_write(moved_grid_1, save_dir + '/' + "rMR-rCT_def-grid_" + name_1 + name_2, framework)
        print('first reg type, number %d', index)

    # running for registration fake_CT <-> real_CT
    temp_dirs_2, ref_dirs_2 = get_dirs(data_root, 'all_fake_CT', 'all_real_CT', is_fake='first', pat=pat)
    if do_val:
        seg_temp_dirs_2, seg_ref_dirs_2 = get_dirs(segm_dir, 'all_real_MR', 'all_real_CT', pat=pat)

    if framework != 'ANTs':
        case = 'case1'

    for index, item in enumerate(temp_dirs_2):
        name_1 = item.rsplit(sep=sep, maxsplit=1)[-1]
        name_1 = cut_extention(name_1)
        name_2 = ref_dirs_2[index].rsplit(sep=sep, maxsplit=1)[-1]

        moved_2, def_2, fixed_2, moved_grid_2 = registrate_vols(index, item, ref_dirs_2[index],save_dir, method, framework, reg_case=case)
        moved_real_2 = apply_def(temp_dirs_1[index], ref_dirs_2[index], def_2, framework, save_dir + '/applied-transfo_vol_%d' % (12+2*index) + '.nii.gz')
        name_real_2 = temp_dirs_1[index].rsplit(sep=sep, maxsplit=1)[-1]
        name_real_2 = cut_extention(name_real_2)

        if do_val:
            if framework == 'corrfield':
                fixed = fixed_2
            dice_scores['fCTrCT'].append(calculate_dice_values(seg_temp_dirs_2[index], seg_ref_dirs_2[index], def_2, framework, save_dir=save_dir, case=case))
            dice_fCTrCT.append(gather_dice(seg_temp_dirs_2[index], seg_ref_dirs_2[index], run, method, image_origin, framework, save_dir, def_2, case=case))
        
        image_write(moved_2, save_dir + '/' + "fCT-rCT_moved_" + name_1 + name_2, framework)
        image_write(fixed_2, save_dir + '/' + "fCT-rCT_fixed_" + name_1 + name_2, framework)
        image_write(moved_real_2, save_dir + '/' + "fCT-rCT_applied2rMR_" + name_real_2 + name_2, framework)
        image_write(moved_grid_2, save_dir + '/' + "fCT-rCT_def-grid_" + name_1 + name_2, framework)
        print('second reg type, number %d', index)
    
    # running for registration real_MR <-> fake_MR
    temp_dirs_3, ref_dirs_3 = get_dirs(data_root, 'all_real_MR', 'all_fake_MR', is_fake='second', pat=pat)
    if do_val:
        seg_temp_dirs_3, seg_ref_dirs_3 = get_dirs(segm_dir, 'all_real_MR', 'all_real_CT', pat=pat)

    if framework != 'ANTs':
        case = 'case2'

    for index, item in enumerate(temp_dirs_3):
        name_1 = item.rsplit(sep=sep, maxsplit=1)[-1]
        name_1 = cut_extention(name_1)
        name_2 = ref_dirs_3[index].rsplit(sep=sep, maxsplit=1)[-1]

        moved_3, def_3, fixed_3, moved_grid_3 = registrate_vols(index, item, ref_dirs_3[index], save_dir, method, framework, reg_case=case)

        if do_val:
            if framework == 'corrfield':
                fixed = fixed_3
            dice_scores['rMRfMR'].append(calculate_dice_values(seg_temp_dirs_3[index], seg_ref_dirs_3[index], def_3, framework, save_dir=save_dir, case=case))
            dice_rMRfMR.append(gather_dice(seg_temp_dirs_3[index], seg_ref_dirs_3[index], run, method, image_origin, framework, save_dir, def_3, case=case))

        image_write(moved_3, save_dir + '/' + "rMR-fMR_moved_" + name_1 + name_2, framework)
        image_write(fixed_3, save_dir + '/' + "rMR-fMR_fixed_" + name_1 + name_2, framework)
        image_write(moved_grid_3, save_dir + '/' + "rMR-fMR_def-grid_" + name_1 + name_2, framework)
        print('third reg type, number %d', index)

    if framework == 'ANTs':
        file = open(os.path.join(save_dir, 'dice_scores_' + method + '.txt'), 'wb')
    else:
        file = open(os.path.join(save_dir, 'dice_scores.txt'), 'wb')
    pickle.dump(dice_scores, file)
    file.close

    if framework == 'ANTs':
        file = open(os.path.join(save_dir, 'dice_scores_slicewise_' + method + '.txt'), 'wb')
    else:
        file = open(os.path.join(save_dir, 'dice_scores_slicewise.txt'), 'wb')
    pickle.dump({'init': dice_init, 'rMRrCT': dice_rMRrCT, 'fCTrCT': dice_fCTrCT, 'rMRfMR': dice_rMRfMR}, file)
    file.close

    write_csv(run, method, image_origin, framework, dice_scores, dice_save_dir, date)
    write_csv_all(run, method, image_origin, framework, {'init': dice_init, 'rMRrCT': dice_rMRrCT, 'fCTrCT': dice_fCTrCT, 'rMRfMR': dice_rMRfMR}, dice_save_dir, date, pat)
    plot_dices(dice_init, dice_rMRrCT, dice_fCTrCT, dice_rMRfMR, save_dir)
