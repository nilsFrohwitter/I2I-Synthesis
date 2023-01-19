"""
This code is highly related to the PyTorch-CycleGAN implementaion
from Zhu et al.:
https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix

"""

from .aligned_dataset import AlignedDataset
from .unaligned_dataset import UnalignedDataset
from .aligned_volume_dataset import AlignedVolumeDataset
from torch.utils.data import DataLoader


class CustomDataLoader:
    """CustomDataLoader contains different datasets/dataloaders for training,
    testing an validating. It also considers differences in used data formats
    (e. g. nii for MRIs or jpg for horse(pferd) images).
    Some of these are hard-coded so be careful.
    """

    def __init__(self, opt, mode, d_type='img'):
        self.dataroot = opt.dataroot
        self.mode = mode
        if self.mode == 'train':
            self.dataset = UnalignedDataset(opt, d_type, mode)
            self.dataloader = DataLoader(
                self.dataset,
                batch_size=opt.batch_size,
                shuffle=True,
                num_workers=0  # num_workers can also be 4 (try out)
            )
        elif self.mode == 'val' and opt.data_a == 'pferd':
            self.dataset = UnalignedDataset(opt, d_type, mode)
            self.dataloader = DataLoader(
                self.dataset,
                batch_size=opt.batch_size,
                shuffle=False,
                num_workers=0  # num_workers can also be 4 (try out)
            )
        elif self.mode == 'val' and opt.val_as_vol:
            self.dataset = AlignedVolumeDataset(opt, d_type, mode)
            self.dataloader = DataLoader(
                self.dataset,
                batch_size=opt.batch_size,
                shuffle=False,
                num_workers=0  # num_workers can also be 4 (try out)
            )
        elif self.mode == 'test' and opt.test_as_vol:
            self.dataset = AlignedVolumeDataset(opt, d_type)
            self.dataloader = DataLoader(
                self.dataset,
                batch_size=opt.batch_size,
                shuffle=False,
                num_workers=0  # num_workers can also be 4 (try out)
            )
        else:
            # val and test as slices not volumes
            self.dataset = AlignedDataset(opt, d_type, mode)
            self.dataloader = DataLoader(
                self.dataset,
                batch_size=opt.batch_size,
                shuffle=False,
                num_workers=0  # num_workers can also be 4 (try out)
            )

    def __getitem__(self, item):
        data = self.dataset[item]
        return data

    def load_data(self):
        return self

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        for i, data in enumerate(self.dataloader):
            yield data


def create_val_dataset(opt):
    data_loader = CustomDataLoader(opt, mode='val')
    dataset = data_loader.load_data()
    return dataset


def create_test_dataset(opt):
    data_loader = CustomDataLoader(opt, mode='test')
    dataset = data_loader.load_data()
    return dataset


def create_train_dataset(opt):
    data_loader = CustomDataLoader(opt, mode='train')
    dataset = data_loader.load_data()
    return dataset
