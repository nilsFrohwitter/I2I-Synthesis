import numpy as np
import nibabel as nib
import os
import csv
from registration.utils import get_dirs
from evaluation.measures import *

"""
Now all possible scores of the input-val images are created here.
These are stored in a table depending on their volume.
Values over the entire volume, as well as over the individual layers are also
created and saved.

What we want to know about the images:
    - SNR (signal-to-noise ratio)
    - Entropy
    - Mean
    - Varianz
    - Contrast per Pixel (cpp)

Regarding synthesized images, we also want to know:
    - SSIM
    - PSNR
    - IS
"""

# Daten laden, sollten eigentlich erstmal nur die 2*8 Volumen sein
# Auch die 8 SynthCT-Volumen und SynthMR-Volumen

"""<runs> contains the 8 LOOCV (Leave-One-Out-Cross-Validation) runs of the 8
validation images. The suffixes of these images are given in <patients>.
For adaptation, one might need to adjust some of this code.
"""

runs = ['22-05-11_10-46-49_e5f90ec4-d106-11ec-ab45-ac1f6bf5ab70',
        '22-05-10_17-24-34_4bc31b56-d075-11ec-a90a-ac1f6bf8ff9c',
        '22-05-10_17-37-31_1afa2fda-d077-11ec-8cbf-ac1f6bf5ab70',
        '22-05-10_17-52-01_21a855d0-d079-11ec-8a52-ac1f6bf5ab70',
        '22-05-11_08-39-01_0af31880-d0f5-11ec-bd41-ac1f6bf8ff9c',
        '22-05-11_10-50-33_6b68bb18-d107-11ec-8c8e-ac1f6bf8ff9c',
        '22-05-11_10-55-42_233c8e04-d108-11ec-b93a-ac1f6bf8ff9c',
        '22-05-11_13-15-14_a183f906-d11b-11ec-a9a9-ac1f6bf8ff9c']
Syn_MR_dirs = []
Syn_CT_dirs = []
for i in range(len(runs)):
    Syn_MR_dir, _ = get_dirs(os.path.join(r'C:\Users\Nils\Documents\sharedfoulderVM\val_images', runs[i]), 'all_fake_MR', 'all_real_MR', is_fake='first')
    _, Syn_CT_dir = get_dirs(os.path.join(r'C:\Users\Nils\Documents\sharedfoulderVM\val_images', runs[i]), 'all_real_CT', 'all_fake_CT', is_fake='second')
    Syn_MR_dirs.append(Syn_MR_dir)
    Syn_CT_dirs.append(Syn_CT_dir)


data_root = r'C:\Users\Nils\Documents\sharedfoulderVM\val_images'
save_dir = data_root
CT_dirs, MR_dirs = get_dirs(data_root, r'val_images\all_real_CT', r'val_images\all_real_MR') 

patients = ['02', '04', '06', '08', '10', '12', '14', '16']
modality = ['CT', 'MR']

for i in range(len(CT_dirs)):
    CT_img = nib.load(CT_dirs[i]).get_fdata()
    MR_img = nib.load(MR_dirs[i]).get_fdata()

    CT_img_min = np.min(CT_img)
    CT_img_max = np.max(CT_img)
    MR_img_min = np.min(MR_img)
    MR_img_max = np.max(MR_img)

    mean_ct, std_ct = statis_info(CT_img, CT_img_min, CT_img_max, is_CT=True)
    snr_ct = 0
    cpp_ct = my_cpp(CT_img)
    entropy_vol_ct = entropy(CT_img, CT_img_min, CT_img_max, is_CT=True)

    mean_mr, std_mr = statis_info(MR_img, MR_img_min, MR_img_max, is_CT=False)
    snr_mr = 0
    cpp_mr = my_cpp(MR_img)
    entropy_vol_mr = entropy(MR_img, MR_img_min, MR_img_max, is_CT=False)

    # now the synthetic images...
    CT_synth_img = nib.load(Syn_CT_dirs[i][0]).get_fdata()
    MR_synth_img = nib.load(Syn_MR_dirs[i][0]).get_fdata()
    
    CT_synth_img_min = np.min(CT_synth_img)
    CT_synth_img_max = np.max(CT_synth_img)
    MR_synth_img_min = np.min(MR_synth_img)
    MR_synth_img_max = np.max(MR_synth_img)

    mean_ct_synth, std_ct_synth = statis_info(CT_synth_img, CT_synth_img_min, CT_synth_img_max, is_CT=True)
    snr_ct_synth = 0
    cpp_ct_synth = my_cpp(CT_synth_img)
    entropy_vol_ct_synth = entropy(CT_synth_img, CT_synth_img_min, CT_synth_img_max, is_CT=True)

    mean_mr_synth, std_mr_synth = statis_info(MR_synth_img, MR_synth_img_min, MR_synth_img_max, is_CT=False)
    snr_mr_synth = 0
    cpp_mr_synth = my_cpp(MR_synth_img)
    entropy_vol_mr_synth = entropy(MR_synth_img, MR_synth_img_min, MR_synth_img_max, is_CT=False)

    paramsDict_slice = {}
    paramsDict_slice['volume'] = patients[i]
    for sl in range(CT_img.shape[-1]):
        img_slice_ct = CT_img[:, :, sl]
        img_slice_mr = MR_img[:, :, sl]

        mean_ct_sl, std_ct_sl = statis_info(img_slice_ct, CT_img_min, CT_img_max, is_CT=True)
        snr_ct_sl = 0
        cpp_ct_sl = my_cpp(img_slice_ct)
        entropy_ct_sl = entropy(img_slice_ct, CT_img_min, CT_img_max, is_CT=True)

        mean_mr_sl, std_mr_sl = statis_info(img_slice_mr, MR_img_min, MR_img_max, is_CT=False)
        snr_mr_sl = 0
        cpp_mr_sl = my_cpp(img_slice_mr)
        entropy_mr_sl = entropy(img_slice_mr, MR_img_min, MR_img_max, is_CT=False)

        # now again the synthetic images...
        img_slice_ct_synth = CT_synth_img[:, :, sl]
        img_slice_mr_synth = MR_synth_img[:, :, sl]

        mean_ct_synth_sl, std_ct_synth_sl = statis_info(img_slice_ct_synth, CT_synth_img_min, CT_synth_img_max, is_CT=True)
        snr_ct_synth_sl = 0
        cpp_ct_synth_sl = my_cpp(img_slice_ct_synth)
        entropy_ct_synth_sl = entropy(img_slice_ct_synth, CT_synth_img_min, CT_synth_img_max, is_CT=True)

        mean_mr_synth_sl, std_mr_synth_sl = statis_info(img_slice_mr_synth, MR_synth_img_min, MR_synth_img_max, is_CT=False)
        snr_mr_synth_sl = 0
        cpp_mr_synth_sl = my_cpp(img_slice_mr_synth)
        entropy_mr_synth_sl = entropy(img_slice_mr_synth, MR_synth_img_min, MR_synth_img_max, is_CT=False)

        paramsDict_slice['slice'] = sl + 1
        paramsDict_slice['mean_ct'] = mean_ct_sl
        paramsDict_slice['std_ct'] = std_ct_sl
        # paramsDict_slice['snr'] = snr_sl
        paramsDict_slice['cpp_ct'] = cpp_ct_sl
        paramsDict_slice['entropy_ct'] = entropy_ct_sl
        paramsDict_slice['mean_mr'] = mean_mr_sl
        paramsDict_slice['std_mr'] = std_mr_sl
        # paramsDict_slice['snr'] = snr_sl
        paramsDict_slice['cpp_mr'] = cpp_mr_sl
        paramsDict_slice['entropy_mr'] = entropy_mr_sl

        paramsDict_slice['mean_ct_syn'] = mean_ct_synth_sl
        paramsDict_slice['std_ct_syn'] = std_ct_synth_sl
        # paramsDict_slice['snr'] = snr_sl
        paramsDict_slice['cpp_ct_syn'] = cpp_ct_synth_sl
        paramsDict_slice['entropy_ct_syn'] = entropy_ct_synth_sl
        paramsDict_slice['mean_mr_syn'] = mean_mr_synth_sl
        paramsDict_slice['std_mr_syn'] = std_mr_synth_sl
        # paramsDict_slice['snr'] = snr_sl
        paramsDict_slice['cpp_mr_syn'] = cpp_mr_synth_sl
        paramsDict_slice['entropy_mr_syn'] = entropy_mr_synth_sl

        fieldnames_slice = ['volume', 'slice', 'mean_ct', 'std_ct', 'cpp_ct', 'entropy_ct',
                            'mean_mr', 'std_mr', 'cpp_mr', 'entropy_mr', 
                            'mean_ct_syn', 'std_ct_syn', 'cpp_ct_syn', 'entropy_ct_syn',
                            'mean_mr_syn', 'std_mr_syn', 'cpp_mr_syn', 'entropy_mr_syn']
        dict_slice_Name = 'volume_statistics_slicewise_new.csv'

        if not os.path.exists(os.path.join(save_dir, dict_slice_Name)):
            with open(os.path.join(save_dir, dict_slice_Name), 'a', newline='') as csvfile:
                CSVwriter = csv.DictWriter(csvfile, fieldnames=fieldnames_slice)
                CSVwriter.writeheader()
        elif i == 0 and sl == 0:
            raise NameError("Be carefull. Don't override the already created statistics file.")

        with open(os.path.join(save_dir, dict_slice_Name), 'a', newline='') as csvfile:
            CSVwriter = csv.DictWriter(csvfile, fieldnames=fieldnames_slice)
            CSVwriter.writerow(paramsDict_slice)

    # now, save everything in a csv
    paramsDict_vol = {}
    paramsDict_vol['volume'] = patients[i]
    paramsDict_vol['mean_ct'] = mean_ct
    paramsDict_vol['std_ct'] = std_ct
    # paramsDict_vol['snr_ct'] = snr_ct
    paramsDict_vol['cpp_ct'] = cpp_ct
    paramsDict_vol['entropy_ct'] = entropy_vol_ct
    paramsDict_vol['mean_mr'] = mean_mr
    paramsDict_vol['std_mr'] = std_mr
    # paramsDict_vol['snr_mr'] = snr_mr
    paramsDict_vol['cpp_mr'] = cpp_mr
    paramsDict_vol['entropy_mr'] = entropy_vol_mr

    paramsDict_vol['mean_ct_syn'] = mean_ct_synth
    paramsDict_vol['std_ct_syn'] = std_ct_synth
    # paramsDict_vol['snr_ct'] = snr_ct
    paramsDict_vol['cpp_ct_syn'] = cpp_ct_synth
    paramsDict_vol['entropy_ct_syn'] = entropy_vol_ct_synth
    paramsDict_vol['mean_mr_syn'] = mean_mr_synth
    paramsDict_vol['std_mr_syn'] = std_mr_synth
    # paramsDict_vol['snr_mr'] = snr_mr
    paramsDict_vol['cpp_mr_syn'] = cpp_mr_synth
    paramsDict_vol['entropy_mr_syn'] = entropy_vol_mr_synth

    fieldnames_vol = ['volume', 'mean_ct', 'std_ct', 'cpp_ct', 'entropy_ct', 
                      'mean_mr', 'std_mr', 'cpp_mr', 'entropy_mr', 
                      'mean_ct_syn', 'std_ct_syn', 'cpp_ct_syn', 'entropy_ct_syn', 
                      'mean_mr_syn', 'std_mr_syn', 'cpp_mr_syn', 'entropy_mr_syn']
    dict_vol_Name = 'volume_statistics_new.csv'

    if not os.path.exists(os.path.join(save_dir, dict_vol_Name)):
        with open(os.path.join(save_dir, dict_vol_Name), 'a', newline='') as csvfile:
            CSVwriter = csv.DictWriter(csvfile, fieldnames=fieldnames_vol)
            CSVwriter.writeheader()

    with open(os.path.join(save_dir, dict_vol_Name), 'a', newline='') as csvfile:
        CSVwriter = csv.DictWriter(csvfile, fieldnames=fieldnames_vol)
        CSVwriter.writerow(paramsDict_vol)
