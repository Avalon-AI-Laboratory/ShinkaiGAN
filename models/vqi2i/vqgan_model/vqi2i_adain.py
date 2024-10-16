import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from models.vqi2i.modules.encoders.network import *
from models.vqi2i.modules.vqvae.quantize import VectorQuantizer
from models.vqi2i.modules.losses.vqlpips import VQLPipsWithDiscriminator
from models.vqi2i.modules.encoders.network import StyleMixingFFN

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
        self.loss_a = VQLPipsWithDiscriminator(perceptual_weight=0, disc_conditional=True)
        self.loss_b = VQLPipsWithDiscriminator(perceptual_weight=0, disc_conditional=True)
        self.loss_c = VQLPipsWithDiscriminator(perceptual_weight=0, disc_conditional=False)

        self.style_enc_a = StyleEncoder(in_channels=in_channels, 
                                        hidden_dimensions=intermediate_channels, 
                                        style_dimensions=style_dim, 
                                        n_downsampling=3)
        
        self.style_enc_b = StyleEncoder(in_channels=in_channels, 
                                        hidden_dimensions=intermediate_channels, 
                                        style_dimensions=style_dim, 
                                        n_downsampling=3)
        
        self.shinkai_style_enc = StyleEncoder(in_channels=in_channels,
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
        
        # self.style_mixer_mlp = nn.Linear(in_features=style_dim*2, out_features=style_dim)
        self.style_mixer_mlp = StyleMixingFFN(style_dim*2, style_dim)
        self.style_mixer_normalization = nn.LayerNorm(style_dim)

    def encode(self, x, label, img_ref=None, style_mix=False, return_content_feats=False):
        if return_content_feats:
            c, feats = self.content_enc(x, extract_feats=True)
        else:
            c = self.content_enc(x)
        c = self.quant_conv(c)
        quant_c, emb_loss, info = self.quantize(c)
        
        # Encode Style
        if label == 1:
            style_vector = self.style_enc_a(x)
        else:
            style_vector = self.style_enc_b(x)
            if style_mix:
                assert img_ref is not None, "Reference Image is required for style mixing"
                s_r = self.shinkai_style_enc(img_ref)
                style_vector = self.style_mix(style_vector, s_r)

        if return_content_feats:
            return quant_c, emb_loss, info, style_vector, feats
            
        return quant_c, emb_loss, info, style_vector
    
    def style_mix(self, style_a, style_b):
        if len(style_a.shape) > 2:
            style_a = style_a.squeeze(-2, -1)
        if len(style_b.shape) > 2:
            style_b = style_b.squeeze(-2, -1)
        style_vector = torch.cat([style_a, style_b], dim=1)
        style_vector = self.style_mixer_mlp(style_vector)
        style_vector = self.style_mixer_normalization(style_vector)
        return style_vector.unsqueeze(-1).unsqueeze(-1)

    def encode_style(self, x, label):
        if label == 1:
            style_vector = self.style_enc_a(x)
        elif label == 0:
            style_vector = self.style_enc_b(x)
        else:
            style_vector = self.shinkai_style_enc(x)
        
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
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None):
        
        super(VQI2ICrossGAN_AdaIN, self).__init__(
            in_channels, intermediate_channels, 
            out_channels, style_dim, z_channels, 
            double_z, n_embed, embed_dim, channel_multipliers, 
            resblock_counts, attn_resolutions, dropout, 
            resolution, n_adaresblock
        )

    def forward(self, x, label, img_ref=None, style_mix=False, cross=False, s_given=False):
        quant, diff, _, style_vector = self.encode(x, label, img_ref, style_mix)

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
