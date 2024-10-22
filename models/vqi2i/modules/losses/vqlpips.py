import torch
import torch.nn as nn
import torch.nn.functional as F

from models.vqi2i.modules.lpips.vgg import VGG19
from models.vqi2i.modules.discriminators.nlayers_disc import NLayerDiscriminator

# Loss functions for VQLPIPS with Discriminator (VQLPipsWithDiscriminator)
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
    def __init__(self, codebook_weight = 1.0, pixel_loss_weight = 1.0,
                 disc_num_layers = 3, disc_in_channels = 3, disc_factor = 0.8, disc_weight = 1.0,
                 perceptual_weight = 1.0, use_actnorm = False, disc_conditional = False,
                 disc_ndf = 64, disc_loss = "hinge", init_weights="vgg19"):
        super(VQLPipsWithDiscriminator, self).__init__()
        self.codebook_weight = codebook_weight
        self.pixel_weight = pixel_loss_weight
        self.perceptual_loss = VGG19(init_weights=init_weights, feature_mode=True).eval()
        self.perceptual_weight = perceptual_weight
        self.discriminator = NLayerDiscriminator(input_nc=disc_in_channels*2 if disc_conditional else disc_in_channels, ndf=disc_ndf, n_layers=disc_num_layers, use_actnorm=use_actnorm)
        
        # Discriminator loss, default is hinge loss
        if disc_loss == "hinge":
            self.disc_loss = hinge_d_loss
        else:
            self.disc_loss = vanilla_d_loss
        
        self.disc_factor = disc_factor
        self.disc_weight = disc_weight
        self.disc_conditional = disc_conditional
    
    # Forward pass for VQLPipsWithDiscriminator
    def forward(self, codebook_loss, inputs, reconstructions, optimizer_idx, fake=None, # optimizer_idx = 0 untuk Generator, sedangkan optimizer_idx = 1 untuk diskriminator
                last_layer=None, cond=None, split="train",
                switch_weight=0.1): 
        rec_loss = torch.abs(inputs.contiguous() - reconstructions.contiguous())
        
        # Calculate perceptual loss, if perceptual_weight > 0
        if self.perceptual_weight > 0:
            p_loss = self.perceptual_loss(inputs.contiguous(), reconstructions.contiguous())
            rec_loss = rec_loss + self.perceptual_weight * p_loss
        
        else:
            p_loss = torch.tensor([0.0])
        
        nll_loss = rec_loss
        nll_loss = torch.mean(nll_loss)

        # If optimizer_idx = 0, then it's the generator
        if optimizer_idx == 0:
            if cond is None:
                assert not self.disc_conditional
                if (fake is None):
                    logits_fake = self.discriminator(reconstructions.contiguous())
                else:
                    logits_fake = self.discriminator(fake.contiguous())
            
            else:
                assert self.disc_conditional
                if (fake is None):
                    logits_fake = self.discriminator(torch.cat((reconstructions.contiguous(), cond), dim=1))
                else:
                    logits_fake = self.discriminator(torch.cat((fake.contiguous(), cond), dim=1))
            
            g_loss = -torch.mean(logits_fake) # Pake wasserstein GAN

            d_weight = torch.tensor(1.0)
            if cond is None:
                logits_fake = self.discriminator(reconstructions.contiguous())
            else:
                logits_fake = self.discriminator(torch.cat((reconstructions.contiguous(), cond), dim=1))
                
            g_rec_loss = -torch.mean(logits_fake)

            loss = 5*nll_loss + switch_weight * (1.0*g_loss + 0.2*g_rec_loss) + self.codebook_weight * codebook_loss.mean()
            log = {"{}/total_loss".format(split): loss.clone().detach().mean(),
                   "{}/quant_loss".format(split): codebook_loss.detach().mean(),
                   "{}/nll_loss".format(split): nll_loss.detach().mean(),
                   "{}/rec_loss".format(split): rec_loss.detach().mean(),
                   "{}/p_loss".format(split): p_loss.detach().mean(),
                   "{}/g_loss".format(split): g_loss.detach().mean(),
                   }
            
            return loss, log

        # If optimizer_idx = 1, then it's the discriminator
        if optimizer_idx == 1:
            if cond is None:
                logits_real = self.discriminator(inputs.contiguous().detach())
                logits_fake = self.discriminator(reconstructions.contiguous().detach())
            else:
                logits_real = self.discriminator(torch.cat((inputs.contiguous().detach(), cond), dim=1))
                logits_fake = self.discriminator(torch.cat((reconstructions.contiguous().detach(), cond), dim=1))
            
            d_loss = self.disc_loss(logits_real, logits_fake)
            log = {"{}/disc_loss".format(split): d_loss.clone().detach().mean(),
                   "{}/logits_real".format(split): logits_real.detach().mean(),
                   "{}/logits_fake".format(split): logits_fake.detach().mean()
                   }

            return d_loss, log
        
