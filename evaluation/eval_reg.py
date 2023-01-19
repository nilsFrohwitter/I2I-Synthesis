import numpy as np
import nibabel as nib
import ants
import SimpleITK as sitk
import itk as itk


def dice(segm_temp, segm_ref):
    """One evaluation is with DICE:
       2 * sum(union(Temp_segm, Ref_segm)) / (sum(Temp_segm) + sum(Ref_segm))
    """
    segm_types = np.asarray([1, 2, 3, 4])  # assumed we have these four segm labels

    dice_scores = []
    for i in range(segm_types.size):
        k = segm_types[i]
        t = np.copy(segm_temp)
        r = np.copy(segm_ref)

        t[t != k] = 0
        r[r != k] = 0

        if not np.any(t):
            if not np.any(r):
                dice = None
                weight = 0
            else:
                dice = 2 * np.sum(np.multiply(t, r)/k**2) / (np.sum(t)/k + np.sum(r)/k + 0.000001)
                # weight = 1
        elif not np.any(r):
            dice = 2 * np.sum(np.multiply(t, r)/k**2) / (np.sum(t)/k + np.sum(r)/k + 0.000001)
            # weight = 1
        else:
            dice = 2 * np.sum(np.multiply(t, r)/k**2) / (np.sum(t)/k + np.sum(r)/k + 0.000001)
            # weight = 2
        dice_scores.append(dice)
    return dice_scores


def calculate_dice_values(seg_temp_dir, seg_ref_dir, def_field, framework, save_dir=None, case=None, slicewise=False):
    """calculates dice values by first applying the deformation on the 
    segmentation masks and then calculating the dice-overlap of all (4) segms.
    
    Depending on the registration framework, different steps are required.
    """
    if framework == 'ANTs':
        seg_t = nib.load(seg_temp_dir)
        seg_t = np.round(seg_t.get_data().astype(float))
        seg_r = nib.load(seg_ref_dir)
        seg_r = np.round(seg_r.get_data().astype(float))

        seg_t_moved = ants.apply_transforms(ants.from_numpy(seg_r), ants.from_numpy(seg_t), def_field, interpolator='genericLabel')
        seg_t_moved = np.round(seg_t_moved.numpy())     # need natural numbers

        if save_dir:
            name = seg_temp_dir.rsplit(sep='/', maxsplit=1)[-1]
            ants.image_write(ants.from_numpy(seg_t_moved), save_dir + '/' + '_' + "moved_" + name)

    elif framework == 'corrfield':
        seg_r = sitk.ReadImage(seg_ref_dir, sitk.sitkFloat32)
        seg_r.SetDirection([1, 0, 0, 0, 1, 0, 0, 0, 1])
        seg_t = sitk.ReadImage(seg_temp_dir, sitk.sitkFloat32)
        seg_t.SetDirection([1, 0, 0, 0, 1, 0, 0, 0, 1])

        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(seg_r);
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        resampler.SetDefaultPixelValue(0.0)
        resampler.SetTransform(def_field)
        seg_t_moved = resampler.Execute(seg_t)
        seg_t_moved.SetDirection([-1, 0, 0, 0, -1, 0, 0, 0, 1])
        seg_t_moved.SetOrigin([0, 0, 0])

        if save_dir:
            name = seg_temp_dir.rsplit(sep='/', maxsplit=1)[-1]
            sitk.WriteImage(seg_t_moved, save_dir + '/' + case + '_' + "moved_" + name)
    
        seg_t_moved = sitk.GetArrayFromImage(seg_t_moved).transpose(1, 2, 0)
        seg_r = sitk.GetArrayFromImage(seg_r).transpose(1, 2, 0)

    elif framework == 'elastix':
        seg_moving = itk.imread(seg_temp_dir, itk.F)
        seg_r = itk.imread(seg_ref_dir, itk.F)

        transform_map = def_field.GetParameterMap(0)
        transform_map['ResampleInterpolator'] = ["FinalNearestNeighborInterpolator"]

        parameter_new = itk.ParameterObject.New()
        parameter_new.SetParameterMap(transform_map)

        seg_t_moved = itk.transformix_filter(
            seg_moving,
            parameter_new)

        if save_dir:
            name = seg_temp_dir.rsplit(sep='/', maxsplit=1)[-1]
            itk.imwrite(seg_t_moved, save_dir + '/' + case + '_' + "moved_" + name)

        seg_t_moved = itk.GetArrayFromImage(seg_t_moved).transpose(1, 2, 0)
        seg_r = itk.GetArrayFromImage(seg_r).transpose(1, 2, 0)

    if slicewise:
        dice_vals = []
        for i in range(seg_t_moved.shape[-1]):
            dice_vals.append(dice(seg_t_moved[:, :, i], seg_r[:, :, i]))
        return dice_vals
    else:
        return dice(seg_t_moved, seg_r)


def calculate_initial_dice_values(seg_temp_dir, seg_ref_dir, slicewise=False):
    seg_t = nib.load(seg_temp_dir)
    seg_t = np.round(seg_t.get_data().astype(float))
    seg_r = nib.load(seg_ref_dir)
    seg_r = np.round(seg_r.get_data().astype(float))

    if slicewise:
        dice_vals = []
        for i in range(seg_t.shape[-1]):
            dice_vals.append(dice(seg_t[:, :, i], seg_r[:, :, i]))
        return dice_vals
    else:
        return dice(seg_t, seg_r)
