import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets.dataset_utils import transform_to_HU
import os


def init_hists(opt, n_volumes):
    # maybe there is an easier way but this works...
    h_real_a = torch.zeros([n_volumes, opt.n_hist_bins])
    h_real_b = h_real_a.clone().detach()
    h_fake_b = h_real_a.clone().detach()
    h_cycled_a = h_real_a.clone().detach()
    h_fake_a = h_real_a.clone().detach()
    h_cycled_b = h_real_a.clone().detach()

    return h_real_a, h_real_b, h_fake_b, h_cycled_a, h_fake_a, h_cycled_b


def matplot_hist(real_a, real_b, fake_b, cycled_a, fake_a, cycled_b, opt, vol, do_plot, left_cut_only=False):
    if left_cut_only:
        min_val = (opt.clip - opt.clip_range) + 2 * opt.hist_cut
        max_val = opt.clip
    else:
        min_val = (opt.clip - opt.clip_range) + opt.hist_cut
        max_val = opt.clip - opt.hist_cut
    x = np.linspace(min_val, max_val, opt.n_hist_bins)  # -1000 is the shift between grayscale and HU values

    # calculate hists for all volumes
    h_real_a = torch.histc(transform_to_HU(real_a, opt).cpu(), bins=opt.n_hist_bins, min=min_val, max=max_val)
    h_real_b = torch.histc(transform_to_HU(real_b, opt).cpu(), bins=opt.n_hist_bins, min=min_val, max=max_val)
    h_fake_b = torch.histc(transform_to_HU(fake_b, opt).cpu(), bins=opt.n_hist_bins, min=min_val, max=max_val)
    h_cycled_a = torch.histc(transform_to_HU(cycled_a, opt).cpu(), bins=opt.n_hist_bins, min=min_val, max=max_val)
    h_fake_a = torch.histc(transform_to_HU(fake_a, opt).cpu(), bins=opt.n_hist_bins, min=min_val, max=max_val)
    h_cycled_b = torch.histc(transform_to_HU(cycled_b, opt).cpu(), bins=opt.n_hist_bins, min=min_val, max=max_val)

    # plot first graph:
    # real CBCT with synthesized fake CT with comparison to real CT; cycled CBCT for additive comparison
    plt.figure(0)
    r_a, = plt.plot(x, h_real_a.numpy(), label='real ' + opt.data_a)   # real_a green
    f_b, = plt.plot(x, h_fake_b.numpy(), label='fake ' + opt.data_b)     # fake_b red
    r_b, = plt.plot(x, h_real_b.numpy(), label='real ' + opt.data_b)     # real_b black
    cy_a, = plt.plot(x, h_cycled_a.numpy(), label='cycled ' + opt.data_a)   # cycled_a dotted green
    plt.grid(True)
    plt.xlabel('HU-Werte bzw. pseudo-HU-Werte')
    plt.ylabel('Absolute Häufigkeit')
    plt.legend(handles=[r_a, f_b, r_b, cy_a])
    if do_plot:
        plt.show()
    else:
        plt.close()

    # plot second graph:
    # real CT with synthesized fake CBCT with comparison to real CBCT; cycled CT for additive comparison
    plt.figure(1)
    r_b, = plt.plot(x, h_real_b.numpy(), label='real ' + opt.data_b)  # real_a green
    f_a, = plt.plot(x, h_fake_a.numpy(), label='fake ' + opt.data_a)  # fake_b red
    r_a, = plt.plot(x, h_real_a.numpy(), label='real ' + opt.data_a)  # real_b black
    cy_b, = plt.plot(x, h_cycled_b.numpy(), label='cycled ' + opt.data_b)  # cycled_a dotted green
    plt.grid(True)
    plt.xlabel('HU-Werte bzw. pseudo-HU-Werte')
    plt.ylabel('Absolute Häufigkeit')
    plt.legend(handles=[r_b, f_a, r_a, cy_b])
    if do_plot:
        plt.show()
    else:
        plt.close()

    return h_real_a, h_real_b, h_fake_b, h_cycled_a, h_fake_a, h_cycled_b


def matplot_hist_avg(h_real_a, h_real_b, h_fake_b, h_cycled_a, h_fake_a, h_cycled_b, opt, left_cut_only=False):
    bin_width = (opt.clip_range - 2 * opt.hist_cut) / opt.n_hist_bins
    if left_cut_only:
        min_val = (opt.clip - opt.clip_range) + 2 * opt.hist_cut
        max_val = opt.clip
    else:
        min_val = (opt.clip - opt.clip_range) + opt.hist_cut
        max_val = opt.clip - opt.hist_cut
    x = np.linspace(min_val, max_val, opt.n_hist_bins)  # -1000 is the shift between grayscale and HU values

    # plot first graph:
    # real CBCT with synthesized fake CT with comparison to real CT; cycled CBCT for additive comparison
    plt.figure(0)
    r_a, = plt.plot(x, torch.mean(h_real_a, dim=0).numpy(), label='real ' + opt.data_a)  # real_a green
    f_b, = plt.plot(x, torch.mean(h_fake_b, dim=0).numpy(), label='fake ' + opt.data_b)  # fake_b red
    r_b, = plt.plot(x, torch.mean(h_real_b, dim=0).numpy(), label='real ' + opt.data_b)  # real_b black
    cy_a, = plt.plot(x, torch.mean(h_cycled_a, dim=0).numpy(), label='cycled ' + opt.data_a)  # cycled_a dotted green
    plt.grid(True)
    plt.xlabel('HU-Werte bzw. pseudo-HU-Werte')
    plt.ylabel('Absolute Häufigkeit')
    plt.legend(handles=[r_a, f_b, r_b, cy_a])
    plt.savefig(os.path.join(opt.save_dir, opt.name, 'test_images', 'fake_' + opt.data_b + '_hist.png'), dpi=1000, bbox_inches='tight')
    plt.savefig(os.path.join(opt.save_dir, opt.name, 'test_images', 'fake_' + opt.data_b + '_hist.pdf'), dpi=1000, bbox_inches='tight')
    plt.show()

    # plot second graph:
    # real CT with synthesized fake CBCT with comparison to real CBCT; cycled CT for additive comparison
    plt.figure(1)
    r_b, = plt.plot(x, torch.mean(h_real_b, dim=0).numpy(), label='real ' + opt.data_b)  # real_b green
    f_a, = plt.plot(x, torch.mean(h_fake_a, dim=0).numpy(), label='fake ' + opt.data_a)  # fake_a red
    r_a, = plt.plot(x, torch.mean(h_real_a, dim=0).numpy(), label='real ' + opt.data_a)  # real_a black
    cy_b, = plt.plot(x, torch.mean(h_cycled_b, dim=0).numpy(), label='cycled ' + opt.data_b)  # cycled_b dotted green
    plt.grid(True)
    plt.xlabel('HU-Werte bzw. pseudo-HU-Werte')
    plt.ylabel('Absolute Häufigkeit')
    plt.legend(handles=[r_b, f_a, r_a, cy_b])
    plt.savefig(os.path.join(opt.save_dir, opt.name, 'test_images', 'fake_' + opt.data_a + '_hist.png'), dpi=1000, bbox_inches='tight')
    plt.savefig(os.path.join(opt.save_dir, opt.name, 'test_images', 'fake_' + opt.data_a + '_hist.pdf'), dpi=1000, bbox_inches='tight')
    plt.show()
