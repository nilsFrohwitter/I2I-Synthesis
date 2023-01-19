import torch.nn as nn
import functools


def get_norm_layer(norm):
    """returns a normalization layer for NNs. For now only allows instance and batch normalization."""
    if norm == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
        use_bias = True
    elif norm == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
        use_bias = False
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm)
    return norm_layer, use_bias


class ResidualBlock(nn.Module):
    """This is one of many ways to implement a residual block."""

    def __init__(self, in_channels, out_channels, norm):
        super(ResidualBlock, self).__init__()
        self.norm_layer, self.use_bias = get_norm_layer(norm)

        self.residual_block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, bias=self.use_bias),
            self.norm_layer(out_channels),
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, bias=self.use_bias),
            self.norm_layer(out_channels),
        )

    def forward(self, x):
        out = x + self.residual_block(x)

        return out


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, position, norm):
        super(ConvBlock, self).__init__()
        self.norm_layer, self.use_bias = get_norm_layer(norm)

        if position == 'front':
            self.conv_block = nn.Sequential(
                nn.ReflectionPad2d(3),
                nn.Conv2d(in_channels, out_channels, kernel_size=7, bias=self.use_bias),
                self.norm_layer(out_channels),
                nn.ReLU(True)
            )
        elif position == 'back':
            self.conv_block = nn.Sequential(
                nn.ReflectionPad2d(3),
                nn.Conv2d(in_channels, out_channels, kernel_size=7, padding=0),
                nn.Tanh()
            )
        else:
            raise NotImplementedError('position [%s] is not implemented or empty, it should be "front" or "back"'
                                      % position)

    def forward(self, x):
        return self.conv_block(x)


class ConvStride2Block(nn.Module):
    def __init__(self, in_channels, out_channels, norm):
        super(ConvStride2Block, self).__init__()
        self.norm_layer, self.use_bias = get_norm_layer(norm)

        self.conv_stride2_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=self.use_bias),
            self.norm_layer(out_channels),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.conv_stride2_block(x)


class ConvTransposeBlock(nn.Module):
    """To be able to train learnable up-convolutional weights."""

    def __init__(self, in_channels, out_channels, norm):
        super(ConvTransposeBlock, self).__init__()
        self.norm_layer, self.use_bias = get_norm_layer(norm)

        self.conv_transpose_block = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1,
                               bias=self.use_bias),
            self.norm_layer(out_channels),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.conv_transpose_block(x)
