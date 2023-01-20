import os


def get_dirs(data_root, name_temp, name_ref, is_fake=None):
    """
    This function walks throu the given dirs to get back all the file names inside
    """
    if 'val_images' in data_root:
        d_r_split = data_root.rsplit(sep=os.sep, maxsplit=1)
        data_root = d_r_split[0]
        if is_fake:
            if is_fake == 'first':
                name_temp = d_r_split[1] + '/val_fake_CT'
            elif is_fake == 'second':
                name_ref = d_r_split[1] + '/val_fake_MR'

    if 'seg' in d_r_split[-1]:
        name_temp = 'seg/MR'
        name_ref = 'seg/CT'

    temp_dir_1 = os.path.join(data_root, name_temp)
    ref_dir_1 = os.path.join(data_root, name_ref)
    temp_dir_list = []
    ref_dir_list = []
    for root, _, fnames in sorted(os.walk(temp_dir_1)):
        for fname in fnames:
            temp_dir_list.append(root + os.sep + fname)
    for root, _, fnames in sorted(os.walk(ref_dir_1)):
        for fname in fnames:
            ref_dir_list.append(root + os.sep + fname)

    temp_dir_list.sort()
    ref_dir_list.sort()

    return temp_dir_list, ref_dir_list
