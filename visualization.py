import numpy as np
import matplotlib.pyplot as plt
import torchvision.utils as utils
import torch


def imshow(img, epoch, mode=None):
    img = img / 2 + 0.5         # unnormalize
    npimg = img.cpu().numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    if mode == 'cycled':
        if epoch:
            plt.title('Epoch %d: First row is MR, second row is CT \n'
                      'From left to right: real, fake, cycled' % epoch)
        else:
            plt.title('Testing: First row is MR, second row is CT \n'
                      'From left to right: real, fake, cycled')
    elif mode == 'diff':
        if epoch:
            plt.title('Epoch %d: \n'
                      'MR_real, MR_fake, MR_diff; \n'
                      'CT_real, CT_fake, CT_diff' % epoch)
        else:
            plt.title('Testing \n'
                      'MR_real, MR_fake, MR_diff; \n'
                      'CT_real, CT_fake, CT_diff')
    plt.show()


def visualize(real_a, fake_b, real_b, fake_a, cycled_a=None, cycled_b=None, epoch=None):
    # calculate difference image of real and fake
    diff_image_a = torch.abs(real_a.squeeze(0) - fake_a.squeeze(0))
    diff_image_b = torch.abs(real_b.squeeze(0) - fake_b.squeeze(0))
    if cycled_a is None:
        grid = utils.make_grid([real_a.squeeze(0), fake_a.squeeze(0), diff_image_a,
                                real_b.squeeze(0), fake_b.squeeze(0), diff_image_b], nrow=3, scale_each=True)
        mode = 'diff'
    else:
        grid = utils.make_grid([real_a.squeeze(0), fake_b.squeeze(0),
                                cycled_a.squeeze(0), real_b.squeeze(0),
                                fake_a.squeeze(0), cycled_b.squeeze(0)], nrow=3, scale_each=True)
        mode = 'cycled'
    imshow(grid, epoch, mode=mode)
