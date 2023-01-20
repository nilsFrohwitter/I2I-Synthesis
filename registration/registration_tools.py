import ants
import numpy as np
import nibabel as nib
import os
import csv
import matplotlib.pyplot as plt
import SimpleITK as sitk
import itk as itk
from evaluation.eval_reg import calculate_dice_values, calculate_initial_dice_values


def image_write(image, save_dir, framework):
    if framework == 'ANTs':
        ants.image_write(image, save_dir)
    elif framework == 'corrfield':
        return
    elif framework == 'elastix':
        if image:
            itk.imwrite(image, save_dir)
    else:
        raise NameError('not implemented')

def registrate_vols(i, moved_dir, fixed_dir, save_dir, method='SyN', framework='elastix', reg_case=None):
    """
    registrate_vols registrates the moving image onto the fixed image with the
    given method and method parameter.
    Output of the function is the deformed moving image and the corresponding 
    deformation field.
    
    For now, only ANTs is in use (and is working).
    """
    if framework == 'elastix':
        temp_dir = '/media/sf_sharedfoulderVM/temp/'
        meta_nib = nib.load("/media/sf_sharedfoulderVM/val_images/data_for_corrfield/val_images/seg/mr/seg0016_tcia_MR.nii.gz")

        moving_image_old = nib.load(moved_dir)
        fixed_image_old = nib.load(fixed_dir)

        moving_np = moving_image_old.get_fdata()
        fixed_np = fixed_image_old.get_fdata()

        name_moving = moved_dir.rsplit('/')[-1]
        name_fixed = fixed_dir.rsplit('/')[-1]

        nib.save(nib.Nifti1Image(moving_np, meta_nib.affine, header=meta_nib.header), temp_dir + name_moving)
        nib.save(nib.Nifti1Image(fixed_np, meta_nib.affine, header=meta_nib.header), temp_dir + name_fixed)
        
        moving = itk.imread(temp_dir + name_moving, itk.F)
        fixed = itk.imread(temp_dir + name_fixed , itk.F)

        # now for the registration
        parameter_object = itk.ParameterObject.New()
        default_parameter_map = parameter_object.GetDefaultParameterMap('bspline', 2, 20.0)
        parameter_object.AddParameterMap(default_parameter_map)

        warped_img, def_transf = itk.elastix_registration_method(
            fixed, moving,
            parameter_object=parameter_object,
            log_to_console=True)

        warped_grid = None
    elif framework == 'ANTs':
        moved = nib.load(moved_dir)
        fixed = nib.load(fixed_dir)

        moved = moved.get_data().astype(float)
        fixed = fixed.get_data().astype(float)

        moved = ants.from_numpy(moved)
        fixed = ants.from_numpy(fixed)

        reg = ants.registration(fixed, moved, method)
        warped_img = reg['warpedmovout']
        def_transf = reg['fwdtransforms']

        warped_grid = ants.create_warped_grid(moved, transform=def_transf, fixed_reference_image=fixed)
    elif framework == 'corrfield':
        # loading the already registrated images (moved, fixed) and the deformation-vectors
        fixed = sitk.ReadImage(fixed_dir, sitk.sitkFloat32)
        warped_img = None
        def_transf = sitk.DisplacementFieldTransform(sitk.ReadImage(os.path.join(save_dir, 'displacement-field_' + reg_case + '_' + str(12 + 2*i) + '.nii.gz'), sitk.sitkVectorFloat64))

        warped_grid = None
    return warped_img, def_transf, fixed, warped_grid


def get_dirs(data_root, name_temp, name_ref, is_fake=None, pat=None):
    """
    This function walks throu the given dirs to get back all the file names inside
    """
    if 'corrfield' in data_root:
        name_temp = 'input/images/moving'
        name_ref = 'input/images/fixed'
    elif 'seg_orig' in data_root:
        print('Im in seg_orig')
        name_temp = 'MR'
        name_ref = 'CT'
    elif 'val_images' in data_root:
        d_r_split = data_root.rsplit(sep='/', maxsplit=1)
        data_root = d_r_split[0]
        if is_fake:
            if is_fake == 'first':
                name_temp = d_r_split[1] + '/val_fake_CT'
            elif is_fake == 'second':
                name_ref = d_r_split[1] + '/val_fake_MR'
        if 'seg' in d_r_split[-1]:
            if pat:
                name_temp = 'all_seg/mr'
                name_ref = 'all_seg/ct'
            else:
                name_temp = 'seg/MR'
                name_ref = 'seg/CT'

    temp_dir = os.path.join(data_root, name_temp)
    ref_dir = os.path.join(data_root, name_ref)
    print(temp_dir)
    temp_dir_list = []
    ref_dir_list = []
    for root, _, fnames in sorted(os.walk(temp_dir)):
        for fname in fnames:
            if pat:
                if pat in fname:
                    print(fname)
                    temp_dir_list.append(root + '/' + fname)
            else:
                print(fname)
                temp_dir_list.append(root + '/' + fname)
    for root, _, fnames in sorted(os.walk(ref_dir)):
        for fname in fnames:
            if pat:
                if pat in fname:
                    ref_dir_list.append(root + '/' + fname)
            else:
                ref_dir_list.append(root + '/' + fname)

    temp_dir_list.sort()
    ref_dir_list.sort()

    return temp_dir_list, ref_dir_list


def cut_extention(file_name):
    split = ['moin', 'moin moin']
    while not len(split) == 1:
        split = file_name.rsplit('.', maxsplit=1)
        file_name = split[0]
    return file_name


def apply_def(m_dir, f_dir, def_field, framework, save_dir=None):
    if framework == 'ANTs':
        moving = nib.load(m_dir)
        fixed = nib.load(f_dir)
        moving = moving.get_data().astype(float)
        fixed = fixed.get_data().astype(float)
        moving = ants.from_numpy(moving)
        fixed = ants.from_numpy(fixed)
        img_deformed = ants.apply_transforms(fixed=fixed , moving=moving, transformlist=def_field)
    elif framework == 'corrfield':
        moving = sitk.ReadImage(m_dir, sitk.sitkFloat32)
        fixed = sitk.ReadImage(f_dir, sitk.sitkFloat32)
        def_field.SetInterpolator(sitk.sitkLinear)
        img_deformed = sitk.Resample(moving, def_field, sitk.sitkLinear, 0.0)
        if save_dir:
            sitk.WriteImage(img_deformed, save_dir)
    elif framework == 'elastix':
        temp_dir = '/media/sf_sharedfoulderVM/temp/'

        moving = itk.imread(temp_dir + m_dir.rsplit('/')[-1], itk.F)
        img_deformed = itk.transformix_filter(moving, def_field)
        if save_dir:
            itk.imwrite(img_deformed, save_dir)
    else:
        raise NameError('not implemented')
    return img_deformed


def write_csv(run, method, image_origin, framework, dice_scores, save_dir, date):
    paramsDict = {}
    paramsDict['name'] = run
    paramsDict['method'] = method
    paramsDict['framework'] = framework
    paramsDict['data'] = image_origin
    paramsDict['date'] = date

    names = [key for key in dice_scores.keys()]
    means = [np.mean(np.asarray(dice_scores[i])) for i in dice_scores.keys()]
    stds = [np.std(np.asarray(dice_scores[i]), ddof=1) for i in dice_scores.keys()]

    paramsDict['dice_m_' + names[0]] = means[0]
    paramsDict['dice_std_' + names[0]] = stds[0]
    paramsDict['dice_m_' + names[1]] = means[1]
    paramsDict['dice_std_' + names[1]] = stds[1]
    paramsDict['dice_m_' + names[2]] = means[2]
    paramsDict['dice_std_' + names[2]] = stds[2]
    paramsDict['dice_m_' + names[3]] = means[3]
    paramsDict['dice_std_' + names[3]] = stds[3]

    fieldnames = ['name', 'method', 'framework', 'data', 'date',
                  'dice_m_' + names[0],
                  'dice_std_' + names[0],
                  'dice_m_' + names[1],
                  'dice_std_' + names[1],
                  'dice_m_' + names[2],
                  'dice_std_' + names[2],
                  'dice_m_' + names[3],
                  'dice_std_' + names[3]]

    dictName = 'regResults_LOOCV_new2.csv'

    if not os.path.exists(os.path.join(save_dir, dictName)):
        with open(os.path.join(save_dir, dictName), 'a', newline='') as csvfile:
            CSVwriter = csv.DictWriter(csvfile, fieldnames=fieldnames)
            CSVwriter.writeheader()

    with open(os.path.join(save_dir, dictName), 'a', newline='') as csvfile:
        CSVwriter = csv.DictWriter(csvfile, fieldnames=fieldnames)
        CSVwriter.writerow(paramsDict)


def write_csv_all(run, method, image_origin, framework, dice_slicewise, save_dir, date, pat):
    # write 4 rows in the csv for each dice value
    # in each row, there are the slicewise and segm-wise dice values

    # segment labels are for example liver, spleen, rkidney, lkidney
    segms = ['liver', 'spleen', 'rkidney', 'lkidney']

    paramsDict = {}
    paramsDict['name'] = run
    paramsDict['method'] = method
    paramsDict['framework'] = framework
    paramsDict['data'] = image_origin
    paramsDict['date'] = date

    paramsDict['volume'] = pat
    paramsDict['organ'] = None
    paramsDict['slice'] = None
    paramsDict['init'] = None
    paramsDict['reg'] = None
    paramsDict['reg with CTSyn'] = None
    paramsDict['reg with MRSyn'] = None

    fieldnames = ['name', 'method', 'framework', 'data', 'date', 'volume', 'organ', 'slice', 'init', 'reg', 'reg with CTSyn', 'reg with MRSyn']

    keys = list(dice_slicewise.keys())

    dictName = 'regResults_LOOCV_slicewise_new2.csv'  # specify a name you want
    if not os.path.exists(os.path.join(save_dir, dictName)):
        with open(os.path.join(save_dir, dictName), 'a', newline='') as csvfile:
            CSVwriter = csv.DictWriter(csvfile, fieldnames=fieldnames)
            CSVwriter.writeheader()

    for i in range(len(dice_slicewise[keys[0]])):
        # i is the volume
        if not pat:
            paramsDict['volume'] = str(2 * i + 12)
        for j in range(len(dice_slicewise[keys[0]][i])):
            # j is the slice
            paramsDict['slice'] = j + 1
            for k in range(len(dice_slicewise[keys[0]][i][j])):
                # k is the specific segment in the image
                paramsDict['organ'] = segms[k]
                paramsDict['init'] = dice_slicewise[keys[0]][i][j][k]
                paramsDict['reg'] = dice_slicewise[keys[1]][i][j][k]
                paramsDict['reg with CTSyn'] = dice_slicewise[keys[2]][i][j][k]
                paramsDict['reg with MRSyn'] = dice_slicewise[keys[3]][i][j][k]
                with open(os.path.join(save_dir, dictName), 'a', newline='') as csvfile:
                    CSVwriter = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    CSVwriter.writerow(paramsDict)

 
def gather_dice(temp_dir, ref_dir, run, method, image_origin, framework, save_dir, def_1=None, case=None):
    """this function is for plotting dice values of two different methods
    against each other"""

    if def_1:
        # if fixed:
        #     dice_vals = calculate_dice_values(temp_dir, ref_dir, def_1, framework, slicewise=True, fixed=fixed, case=case)
        # else:
        dice_vals = calculate_dice_values(temp_dir, ref_dir, def_1, framework, slicewise=True, case=case)
    else:
        dice_vals = calculate_initial_dice_values(temp_dir, ref_dir, slicewise=True)

    return dice_vals


def plot_dices(d_inits, d_rs, d_fCTs, d_fMRs, save_dir):
    """saving some plots for dice-comparison. Can also be used to calculate
    outlyers and save them with their specific values."""

    print('first: ' + str(np.asarray(d_inits, dtype=object).shape))
    for i in range(len(d_inits)):
        print('ith part: ' + str(np.asarray(d_inits[i]).shape))
        d_init = np.transpose(d_inits[i])
        d_r = np.transpose(d_rs[i])
        d_fCT = np.transpose(d_fCTs[i])
        print('ith part after transpo: ' + str(np.asarray(d_init).shape))

        # formula to calculate the direct distance from the dice_pairs to the
        # one-to-one map (as a function f(x)=x, x in [0,1]) is sqrt(3)/2*abs(x-y)
        diff_init_real = np.sqrt(3) / 2 * (np.asarray(d_init) - np.asarray(d_r))
        np.save(os.path.join(save_dir, 'diff_init_real_' + "%02d" % (i,)), diff_init_real)

        diff_init_fakeCT = np.sqrt(3) / 2 * (np.asarray(d_init) - np.asarray(d_fCT))
        np.save(os.path.join(save_dir, 'diff_init_fakeCT_' + "%02d" % (i,)), diff_init_fakeCT)

        diff_real_fakeCT = np.sqrt(3) / 2 * (np.asarray(d_r) - np.asarray(d_fCT))
        np.save(os.path.join(save_dir, 'diff_real_fakeCT_' + "%02d" % (i,)), diff_real_fakeCT)

        if True:
            # plotting d_init against d_r
            plt.figure(1)
            plt.subplot(221)
            plt.plot(d_init[0], d_r[0], 'bx')
            plt.title('first')
            plt.xlabel('init')
            plt.ylabel('real')
            plt.plot([0, 1], [0, 1], 'r')
            plt.xlim(0, 1)
            plt.ylim(0, 1)
            plt.axis('square')
            plt.subplot(222)
            plt.plot(d_init[1], d_r[1], 'bx')
            plt.title('second')
            plt.xlabel('init')
            plt.ylabel('real')
            plt.plot([0, 1], [0, 1], 'r')
            plt.xlim(0, 1)
            plt.ylim(0, 1)
            plt.axis('square')
            plt.subplot(223)
            plt.plot(d_init[2], d_r[2], 'bx')
            plt.title('third')
            plt.xlabel('init')
            plt.ylabel('real')
            plt.plot([0, 1], [0, 1], 'r')
            plt.xlim(0, 1)
            plt.ylim(0, 1)
            plt.axis('square')
            plt.subplot(224)
            plt.plot(d_init[3], d_r[3], 'bx')
            plt.title('forth')
            plt.xlabel('init')
            plt.ylabel('real')
            plt.plot([0, 1], [0, 1], 'r')
            plt.xlim(0, 1)
            plt.ylim(0, 1)
            plt.axis('square')
            plt.tight_layout()

            plt.savefig(os.path.join(save_dir, 'dice_values-init_vs_real-volume_' + "%02d" % (i,) + '.png'))
            plt.close()

            # plotting d_init against d_fCT
            plt.figure(2)
            plt.subplot(221)
            plt.plot(d_init[0], d_fCT[0], 'bx')
            plt.title('first')
            plt.xlabel('init')
            plt.ylabel('fCT')
            plt.plot([0, 1], [0, 1], 'r')
            plt.xlim(0, 1)
            plt.ylim(0, 1)
            plt.axis('square')
            plt.subplot(222)
            plt.plot(d_init[1], d_fCT[1], 'bx')
            plt.title('second')
            plt.xlabel('init')
            plt.ylabel('fCT')
            plt.plot([0, 1], [0, 1], 'r')
            plt.xlim(0, 1)
            plt.ylim(0, 1)
            plt.axis('square')
            plt.subplot(223)
            plt.plot(d_init[2], d_fCT[2], 'bx')
            plt.title('third')
            plt.xlabel('init')
            plt.ylabel('fCT')
            plt.plot([0, 1], [0, 1], 'r')
            plt.xlim(0, 1)
            plt.ylim(0, 1)
            plt.axis('square')
            plt.subplot(224)
            plt.plot(d_init[3], d_fCT[3], 'bx')
            plt.title('forth')
            plt.xlabel('init')
            plt.ylabel('fCT')
            plt.plot([0, 1], [0, 1], 'r')
            plt.xlim(0, 1)
            plt.ylim(0, 1)
            plt.axis('square')
            plt.tight_layout()

            plt.savefig(os.path.join(save_dir, 'dice_values-init_vs_fakeCT-volume_' + "%02d" % (i,) + '.png'))
            plt.close()

            # plotting d_r against d_fCT
            plt.figure(3)
            plt.subplot(221)
            plt.plot(d_r[0], d_fCT[0], 'bx')
            plt.title('first')
            plt.xlabel('real')
            plt.ylabel('fCT')
            plt.plot([0, 1], [0, 1], 'r')
            plt.xlim(0, 1)
            plt.ylim(0, 1)
            plt.axis('square')
            plt.subplot(222)
            plt.plot(d_r[1], d_fCT[1], 'bx')
            plt.title('second')
            plt.xlabel('real')
            plt.ylabel('fCT')
            plt.plot([0, 1], [0, 1], 'r')
            plt.xlim(0, 1)
            plt.ylim(0, 1)
            plt.axis('square')
            plt.subplot(223)
            plt.plot(d_r[2], d_fCT[2], 'bx')
            plt.title('third')
            plt.xlabel('real')
            plt.ylabel('fCT')
            plt.plot([0, 1], [0, 1], 'r')
            plt.xlim(0, 1)
            plt.ylim(0, 1)
            plt.axis('square')
            plt.subplot(224)
            plt.plot(d_r[3], d_fCT[3], 'bx')
            plt.title('forth')
            plt.xlabel('real')
            plt.ylabel('fCT')
            plt.plot([0, 1], [0, 1], 'r')
            plt.xlim(0, 1)
            plt.ylim(0, 1)
            plt.axis('square')
            plt.tight_layout()

            plt.savefig(os.path.join(save_dir, 'dice_values-realMR_vs_fakeCT-volume_' + "%02d" % (i,) + '.png'))
            plt.close()
