import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from models.vqi2i.modules.encoders.network import *
from models.vqi2i.modules.vqvae.quantize import VectorQuantizer
# from models.vqi2i.modules.discriminators.model import NLayerDiscriminator
# from models.vqi2i.modules.losses.vqperceptual import hinge_d_loss

class VQI2I_AdaIN(nn.Module):
    def __init__(self,
                 in_channels : int = 3,
                 intermediate_channels : int = 128,
                 out_channels : int = 3,
                 style_dim : int = 128,
                 z_channels : int = 128,
                 double_z : bool = True,
                 n_embed : int = 512,
                 embed_dim : int = 512,
                 channel_multipliers : list = [1,1,2,4,8],
                 resblock_counts : int = 2,
                 attn_resolutions : list = [16],
                 dropout : float = 0.1,
                 resolution : int = 256,
                 n_adaresblock : int = 4,
                 ckpt_path : str = None,
                 ignore_keys : list = [],
                 image_key : str = "image",
                 colorize_nlabels=None,
                 monitor=None
                ):
        
        super(VQI2I_AdaIN, self).__init__()
        self.image_key = image_key

        self.style_enc_a = StyleEncoder(in_channels=in_channels, 
                                        hidden_dimensions=intermediate_channels, 
                                        style_dimensions=style_dim, 
                                        n_downsampling=3)
        
        self.style_enc_b = StyleEncoder(in_channels=in_channels, 
                                        hidden_dimensions=intermediate_channels, 
                                        style_dimensions=style_dim, 
                                        n_downsampling=3)
        
        self.content_enc = ContentEncoder(in_channels=in_channels, 
                                          intermediate_channels=intermediate_channels, 
                                          channel_multipliers=channel_multipliers, 
                                          resblock_counts=resblock_counts, 
                                          attn_resolutions=attn_resolutions, 
                                          dropout=dropout, 
                                          resolution=resolution,
                                          z_channels=z_channels, 
                                          double_z=double_z)
        
        self.quantize = VectorQuantizer(n_e=n_embed, e_dim=embed_dim, beta=0.25)

        self.quant_conv = nn.Conv2d(z_channels * 2 if double_z else z_channels, embed_dim, kernel_size=1, stride=1)
        self.post_quant_conv = nn.Conv2d(embed_dim, z_channels * 2 if double_z else z_channels, kernel_size=1, stride=1)

        self.decoder_a = Decoder(out_channels=out_channels, 
                                 intermediate_channels=intermediate_channels, 
                                 channel_multipliers=channel_multipliers, 
                                 resblock_counts=resblock_counts, 
                                 attn_resolutions=attn_resolutions, 
                                 dropout=dropout, 
                                 resolution=resolution,
                                 z_channels=z_channels, 
                                 n_adaresblock=n_adaresblock, 
                                 style_dim=style_dim, 
                                 double_z=double_z)
        
        self.decoder_b = Decoder(out_channels=out_channels, 
                                 intermediate_channels=intermediate_channels, 
                                 channel_multipliers=channel_multipliers, 
                                 resblock_counts=resblock_counts, 
                                 attn_resolutions=attn_resolutions, 
                                 dropout=dropout, 
                                 resolution=resolution,
                                 z_channels=z_channels, 
                                 n_adaresblock=n_adaresblock, 
                                 style_dim=style_dim, 
                                 double_z=double_z)

    def encode(self, x, label):
        c = self.content_enc(x)
        c = self.quant_conv(c)
        quant_c, emb_loss, info = self.quantize(c)
        
        # Encode Style
        if label == 1:
            style_vector = self.style_enc_a(x)
        else:
            style_vector = self.style_enc_b(x)
        
        return quant_c, emb_loss, info, style_vector
    
    def encode_style(self, x, label):
        if label == 1:
            style_vector = self.style_enc_a(x)
        else:
            style_vector = self.style_enc_b(x)
        
        return style_vector
    
    def encode_content(self, x):
        c = self.content_enc(x)
        c = self.quant_conv(c)
        quant_c, emb_loss, info = self.quantize(c)
        return c, quant_c
    
    def decode_a(self, quant_c, style_vector):
        c = self.post_quant_conv(quant_c)
        x_hat = self.decoder_a(c, style_vector)
        return x_hat
    
    def decode_b(self, quant_c, style_vector):
        c = self.post_quant_conv(quant_c)
        x_hat = self.decoder_b(c, style_vector)
        return x_hat
    
    def forward(self, x, label):
        if label == 1:
            quant_c, diff, _, style_vector = self.encode(x, label)
            dec = self.decode_a(quant_c, style_vector)
        
        else:
            quant_c, diff, _, style_vector = self.encode(x, label)
            dec = self.decode_b(quant_c, style_vector)
        
        return dec, diff

    def get_last_layer(self, label):
        if label == 1:
            return self.decoder_a.conv_out.weight
        else:
            return self.decoder_b.conv_out.weight

class VQI2ICrossGAN_AdaIN(VQI2I_AdaIN):
    def __init__(self,
                 n_embed,
                 embed_dim,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None):
        
        super(VQI2ICrossGAN_AdaIN, self).__init__(
            n_embed=n_embed, embed_dim=embed_dim
        )
    
    def forward(self, x, label, cross=False, s_given=False):
        quant, diff, _, style_vector = self.encode(x, label)

        if label == 1:
            if cross == False:
                output = self.decode_a(quant, style_vector)
            else:
                style_vector = s_given
                output = self.decode_b(quant, style_vector)
        else:
            if cross == False:
                output = self.decode_b(quant, style_vector)
            else:
                style_vector = s_given
                output = self.decode_a(quant, style_vector)
        
        return output, diff, style_vector
