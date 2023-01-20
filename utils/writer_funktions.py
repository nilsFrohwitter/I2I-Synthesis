import torch


def write_hists(writer, histograms, opt, n_data=128):
    for i in range(opt.n_hist_bins):
        writer.add_scalar('hist_' + opt.data_b + '/cycled', histograms['cycled_B'][i]/n_data, i)
        writer.add_scalar('hist_' + opt.data_b + '/fake', histograms['fake_B'][i]/n_data, i)
        writer.add_scalar('hist_' + opt.data_b + '/real', histograms['real_B'][i]/n_data, i)
        writer.add_scalar('hist_' + opt.data_a + '/cycled', histograms['cycled_A'][i]/n_data, i)
        writer.add_scalar('hist_' + opt.data_a + '/fake', histograms['fake_A'][i]/n_data, i)
        writer.add_scalar('hist_' + opt.data_a + '/real', histograms['real_A'][i]/n_data, i)
    histograms['real_B'] = torch.zeros([opt.n_hist_bins])
    histograms['fake_B'] = torch.zeros([opt.n_hist_bins])
    histograms['cycled_B'] = torch.zeros([opt.n_hist_bins])
    histograms['real_A'] = torch.zeros([opt.n_hist_bins])
    histograms['fake_A'] = torch.zeros([opt.n_hist_bins])
    histograms['cycled_A'] = torch.zeros([opt.n_hist_bins])
    return histograms


def write_hist(writer, epoch, opt, vol_a, vol_b, vol_results, val_size):
    """plots a original histrogram for one specific volume over the training epochs"""
    bins = 'sturges'    # np.arange(-1, 1, 0.01), 'auto'
    writer.add_histogram('hist/cycled_A', vol_results['h_c_a'][:, :, :val_size], epoch, bins=bins)
    writer.add_histogram('hist/fake_B', vol_results['h_f_b'][:, :, :val_size], epoch, bins=bins)
    writer.add_histogram('hist/real_A', vol_a, epoch, bins=bins)
    writer.add_histogram('hist/cycled_B', vol_results['h_c_b'][:, :, :val_size], epoch, bins=bins)
    writer.add_histogram('hist/fake_A', vol_results['h_f_a'][:, :, :val_size], epoch, bins=bins)
    writer.add_histogram('hist/real_B', vol_b, epoch, bins=bins)


def write_final_hist(writer, histograms, opt, n_data=128):
    for i in range(opt.n_hist_bins):
        writer.add_scalar('hist_' + opt.data_b + '/cycled', histograms['cycled_B'][i] / n_data, i)
        writer.add_scalar('hist_' + opt.data_b + '/fake', histograms['fake_B'][i] / n_data, i)
        writer.add_scalar('hist_' + opt.data_b + '/real', histograms['real_B'][i] / n_data, i)
        writer.add_scalar('hist_' + opt.data_a + '/cycled', histograms['cycled_A'][i] / n_data, i)
        writer.add_scalar('hist_' + opt.data_a + '/fake', histograms['fake_A'][i] / n_data, i)
        writer.add_scalar('hist_' + opt.data_a + '/real', histograms['real_A'][i] / n_data, i)


def write_hist_line(writer, histograms, epoch, opt, n_data=128):
    """plotting the line plot histrogram of one volume containing of n_data slices.
    Overall creates 4 plots, (fake_A, fake_B, cycle_A, cycle_B) """
    epoch_string = format(epoch, '02d')
    for i in range(opt.n_hist_bins):
        writer.add_scalar('hist_' + opt.data_b + '/cycled/epoch_' + epoch_string, histograms['cycled_B'][i] / n_data, i)
        writer.add_scalar('hist_' + opt.data_b + '/fake/epoch_' + epoch_string, histograms['fake_B'][i] / n_data, i)
        writer.add_scalar('hist_' + opt.data_a + '/cycled/epoch_' + epoch_string, histograms['cycled_A'][i] / n_data, i)
        writer.add_scalar('hist_' + opt.data_a + '/fake/epoch_' + epoch_string, histograms['fake_A'][i] / n_data, i)
    if not epoch == (opt.n_epochs + opt.n_epochs_decay - 1):
        histograms['real_B'] = torch.zeros([opt.n_hist_bins])
        histograms['fake_B'] = torch.zeros([opt.n_hist_bins])
        histograms['cycled_B'] = torch.zeros([opt.n_hist_bins])
        histograms['real_A'] = torch.zeros([opt.n_hist_bins])
        histograms['fake_A'] = torch.zeros([opt.n_hist_bins])
        histograms['cycled_A'] = torch.zeros([opt.n_hist_bins])
