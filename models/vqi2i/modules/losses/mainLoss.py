#### MASIH ON PROGRESS... ####

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.vqi2i.modules.lpips.vgg import VGG19
from models.vqi2i.modules.discriminators.nlayers_disc import NLayerDiscriminator

def hinge_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.relu(1. - logits_real))
    loss_fake = torch.mean(F.relu(1. + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss

def vanilla_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.softplus(-logits_real))
    loss_fake = torch.mean(F.softplus(logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss

class VQLPipsWithDiscriminator(nn.Module):
    def __init__(self, disc_start, codebook_weight = 1.0, pixel_loss_weight = 1.0,
                 disc_num_layers = 3, disc_in_channels = 3, disc_factor = 0.8, disc_weight = 1.0,
                 perceptual_weight = 1.0, use_actnorm = False, disc_conditional = False,
                 disc_ndf = 64, disc_loss = "hinge"):
        super(VQLPipsWithDiscriminator, self).__init__()
        self.codebook_weight = codebook_weight
        self.pixel_weight = pixel_loss_weight
        self.perceptual_loss = VGG19.eval(init_weights=True, feature_mode=True)
        self.perceptual_weight = perceptual_weight
        self.discriminator = NLayerDiscriminator(input_nc=disc_in_channels, ndf=disc_ndf, n_layers=disc_num_layers, use_actnorm=use_actnorm)
        
        if disc_loss == "hinge":
            self.disc_loss = hinge_d_loss
        else:
            self.disc_loss = vanilla_d_loss
        
        self.disc_factor = disc_factor
        self.disc_weight = disc_weight
        self.disc_conditional = disc_conditional
    
    
        
