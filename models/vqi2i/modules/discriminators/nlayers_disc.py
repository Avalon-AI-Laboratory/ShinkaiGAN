# Kode ini mendefinisikan arsitektur discriminator berbasis PatchGAN yang digunakan dalam model generatif. 
# Discriminator ini bertugas untuk membedakan antara gambar asli dan yang dihasilkan. 
# Fungsi inisialisasi bobot memastikan bahwa model dimulai dengan bobot yang baik, dan implementasi normalisasi 
# baik melalui InstanceNorm maupun ActNorm membantu dalam stabilitas pelatihan. Metode forward menyediakan aliran standar untuk pemrosesan input melalui jaringan.

import functools
import torch
import torch.nn as nn

def weights_init(m):
    classname = m.__class__.__name__
    # if conv layer, initialize with normal distribution
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    # if batchnorm layer, initialize with normal distribution and bias with 0   
    elif classname.find('InstanceNorm') != -1:
    # elif classname.find('LayerNorm') != -1:
    # elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


# The discriminator is defined as a PatchGAN discriminator as in Pix2Pix
# It is a convolutional neural network that is used to classify whether a patch of an image is real or fake
class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator as in Pix2Pix
        --> see https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
    """
    def __init__(self, input_nc=3, ndf=64, n_layers=3, use_actnorm=False):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        # Normalization layer
        super(NLayerDiscriminator, self).__init__()
        if not use_actnorm:
            # norm_layer = nn.BatchNorm2d
            norm_layer = nn.InstanceNorm2d
            # norm_layer = nn.LayerNorm
        else:
            norm_layer = ActNorm
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            # use_bias = norm_layer.func != nn.BatchNorm2d
            use_bias = norm_layer.func != nn.InstanceNorm2d
            # use_bias = norm_layer.func != nn.LayerNorm
        else:
            # use_bias = norm_layer != nn.BatchNorm2d
            use_bias = norm_layer != nn.InstanceNorm2d
            # use_bias = norm_layer != nn.LayerNorm

        # Build the discriminator
        kw = 4 # kernel size convolution (4x4)
        padw = 1 # padding size convolution (1x1)
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        # Loop through the number of layers. The first layer has 1 filter, and the number of filters doubles for each subsequent layer
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)

            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult, affine=True),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult, affine=True),
            nn.LeakyReLU(0.2, True)
        ]

        # The last layer of the discriminator is a convolutional layer that outputs a 1-channel prediction map
        sequence += [
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.main = nn.Sequential(*sequence)

    # Forward pass of the discriminator
    def forward(self, input):
        """Standard forward."""
        return self.main(input)

# ActNorm layer for feature normalization
class ActNorm(nn.Module):
    def __init__(self, num_features, logdet=False, affine=True,
                 allow_reverse_init=False):
        assert affine
        super().__init__()
        self.logdet = logdet
        self.loc = nn.Parameter(torch.zeros(1, num_features, 1, 1)) # mean of the distribution
        self.scale = nn.Parameter(torch.ones(1, num_features, 1, 1)) # standard deviation of the distribution
        self.allow_reverse_init = allow_reverse_init

        # Initialize the scale parameter to be zero. It saves the state of the layer
        self.register_buffer('initialized', torch.tensor(0, dtype=torch.uint8))
