import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
import numpy as np

def nonlinearity(x):
    # swish
    return x*torch.sigmoid(x)

def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)

class StyleMixingFFN(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.2):
        super(StyleMixingFFN, self).__init__()
        self.fc1 = nn.Linear(in_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, out_dim)

        self.act = nn.ReLU()

        self.dropout = nn.Dropout(dropout)

        self.bn1 = nn.LayerNorm(512)
        self.bn2 = nn.LayerNorm(256)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = self.act(x)
        x = self.dropout(x)

        x = self.fc3(x)

        return x

class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super(Downsample, self).__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = nn.Conv2d(in_channels,
                                  in_channels,
                                  kernel_size=3,
                                  stride=2,
                                  padding=0)

    def forward(self, x):
        if self.with_conv:
            pad = (0,1,0,1)
            x = F.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = F.avg_pool2d(x, kernel_size=2, stride=2)
        return x

class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super(Upsample, self).__init__()
        self.with_conv = with_conv
        
        if self.with_conv:
            self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, stride=1)
    
    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='nearest')

        if self.with_conv:
            x = self.conv(x)
        
        return x

class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None, time_embedding_channel = 512, dropout = 0.1):
        super(ResNetBlock, self).__init__()
        out_channels = out_channels if out_channels is not None else in_channels
        self.norm1 = Normalize(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1)
        self.time_embedding_projection = nn.Linear(time_embedding_channel, out_channels) if time_embedding_channel > 0 else None
        self.norm2 = Normalize(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1)
        self.conv_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0) if in_channels != out_channels else None
    
    def forward(self, x, time_embedding=None):
        h = self.norm1(x)
        h = nonlinearity(h)
        h = self.conv1(h)
        if time_embedding is not None:
            h += self.time_embedding_projection(nonlinearity(time_embedding))[:, :, None, None]
        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)
        if self.conv_shortcut is not None:
            x = self.conv_shortcut(x)
        
        return x + h

class StyleEncoder(nn.Module):
    def __init__(self, in_channels, hidden_dimensions,
                 style_dimensions, n_downsampling):
        super(StyleEncoder, self).__init__()
        self.use_bias = True
        self.in_channels = in_channels
        self.hidden_dimensions = hidden_dimensions
        self.style_dimensions = style_dimensions
        self.n_downsampling = n_downsampling
        self.block1 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels=in_channels, out_channels=hidden_dimensions, kernel_size=7, stride=1, bias=self.use_bias),
            nn.ReLU()
        )
        self.block2 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=hidden_dimensions, out_channels=hidden_dimensions*2, kernel_size=4, stride=2, bias=self.use_bias),
            nn.ReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=hidden_dimensions*2, out_channels=hidden_dimensions*4, kernel_size=4, stride=2, bias=self.use_bias),
            nn.ReLU()
        )
        self.block3 = []
        
        for i in range(self.n_downsampling - 2):
            self.block3 += [
                nn.ReflectionPad2d(1), 
                nn.Conv2d(in_channels=hidden_dimensions*4, out_channels=hidden_dimensions*4, kernel_size=4, stride=2, bias=self.use_bias),
                nn.ReLU()
            ]
        
        self.block3 = nn.Sequential(*self.block3)

        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.ffn = nn.Conv2d(hidden_dimensions*4, self.style_dimensions, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.global_pooling(x)
        x = self.ffn(x)
        return x

class LinearAttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super(LinearAttentionBlock, self).__init__()
        self.norm = nn.BatchNorm2d(in_channels)  # Using batch normalization for simplicity
        self.Q = nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0, stride=1)
        self.K = nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0, stride=1)
        self.V = nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0, stride=1)
        self.out = nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0, stride=1)

    def forward(self, x):
        x = self.norm(x)
        
        Q = self.Q(x)
        K = self.K(x)
        V = self.V(x)
        
        b, c, h, w = Q.shape
        Q = Q.view(b, c, -1)  # b x c x (h*w)
        K = K.view(b, c, -1)  # b x c x (h*w)
        V = V.view(b, c, -1)  # b x c x (h*w)
        
        K = F.softmax(K, dim=-1)

        attention = torch.bmm(Q, K.transpose(1, 2)) / (c ** 0.5)  # b x (h*w) x (h*w)
        
        out = torch.bmm(attention, V)  # b x (h*w) x (h*w)
        out = out.view(b, c, h, w)
        out = self.out(out)
        
        return out

class ContentEncoder(nn.Module):
    def __init__(self, 
                 in_channels : int,
                 intermediate_channels : int,
                 channel_multipliers : list,
                 resblock_counts : int,
                 attn_resolutions : list,
                 dropout : float = 0.0,
                 resample_with_conv : bool = True,
                 resolution : int = 256,
                 z_channels : int = 256,
                 double_z : bool = True):
        super(ContentEncoder, self).__init__()
        self.in_channels = in_channels
        self.intermediate_channels = intermediate_channels
        self.time_embedding_channels = 0
        self.num_resolutions = len(channel_multipliers)
        self.resblock_counts = resblock_counts
        self.resolution = resolution

        self.conv_in = nn.Conv2d(in_channels=in_channels, 
                                 out_channels=self.intermediate_channels,
                                 kernel_size=3,
                                 stride=1,
                                 padding=1)
        
        current_resolution = resolution
        in_channels_multiplier = (1, ) + tuple(channel_multipliers)
        self.in_channels_multipliers = in_channels_multiplier
        self.down = nn.ModuleList()

        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = intermediate_channels * in_channels_multiplier[i_level]
            block_out = intermediate_channels * channel_multipliers[i_level]

            for i_block in range(self.resblock_counts):
                block.append(
                    ResNetBlock(in_channels=block_in,
                                out_channels=block_out,
                                time_embedding_channel=self.time_embedding_channels,
                                dropout=dropout)
                )
                block_in = block_out
                if current_resolution in attn_resolutions:
                    attn.append(
                        LinearAttentionBlock(in_channels=block_out)
                    )
            
            downblock = nn.Module()
            downblock.block = block
            downblock.attn = attn

            if i_level != self.num_resolutions - 1:
                downblock.downsample = Downsample(block_out, with_conv=resample_with_conv)
                current_resolution = current_resolution // 2
            
            self.down.append(downblock)
        
        self.mid = nn.Module()
        self.mid.block_1 = ResNetBlock(in_channels=block_in, out_channels=block_in, time_embedding_channel=self.time_embedding_channels, dropout=dropout)
        self.mid.attn = LinearAttentionBlock(in_channels=block_in)
        self.mid.block_2 = ResNetBlock(in_channels=block_in, out_channels=block_in, time_embedding_channel=self.time_embedding_channels, dropout=dropout)

        self.norm_out = Normalize(block_in)
        self.conv_out = nn.Conv2d(block_in, 2 * z_channels if double_z else z_channels, kernel_size=3, stride=1, padding=1)
    
    def forward(self, x, temb=None, extract_feats=False, layers_extracted=None):
        feats = [self.conv_in(x)]

        for i_level in range(self.num_resolutions):
            for i_block in range(self.resblock_counts):
                feature = self.down[i_level].block[i_block](feats[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    feature = self.down[i_level].attn[i_block](feature)
                
                feats.append(feature)
            
            if i_level != self.num_resolutions - 1:
                feats.append(self.down[i_level].downsample(feats[-1]))
        
        feature = feats[-1]
        feature = self.mid.block_1(feature, temb)
        feature = self.mid.attn(feature)
        feature = self.mid.block_2(feature, temb)

        feature = self.norm_out(feature)
        feature = nonlinearity(feature)
        feature = self.conv_out(feature)

        if layers_extracted is not None and extract_feats:
            feats_temp = []
            if len(layers_extracted) == 0:
                feats_temp = feats

            else:
                for i in layers_extracted:
                    feats_temp.append(feats[i])
                feats = feats_temp

            return feature, feats
        
        return feature

class Decoder(nn.Module):
    def __init__(self, 
                 out_channels : int, 
                 intermediate_channels : int, 
                 channel_multipliers: list,
                 resblock_counts : int,
                 attn_resolutions : list,
                 dropout : int = 0.0,
                 resolution : int = 256,
                 n_adaresblock : int = 4,
                 style_dim: int = 128,
                 z_channels : int = 256,
                 double_z : bool = True,
                 give_pre_end : bool = False,
                 tanh_out : bool = False):
        super(Decoder, self).__init__()
        self.out_channels = out_channels
        self.intermediate_channels = intermediate_channels
        self.time_embedding_channels = 0
        self.num_resolutions = len(channel_multipliers)
        self.resblock_counts = resblock_counts
        self.resolution = resolution
        self.give_pre_end = give_pre_end
        self.tanh_out = tanh_out
        self.n_adaresblock = n_adaresblock

        block_in = intermediate_channels * channel_multipliers[self.num_resolutions - 1]
        current_resolution = resolution // (2 ** (self.num_resolutions - 1))
        self.z_shape = (1, z_channels, current_resolution, current_resolution) # Ini keknya perlu diperbaiki, ukurannya seharusnya (1, z_channels, current_resolution(dimensi mel), downsampled_timestep (gonna do something about this))
        print("Melakukan operasi pada z dengan dimensi {} = {} dimensi.".format(
              self.z_shape, np.prod(self.z_shape)))

        # z masuk ke dalam decoder, mulai dari conv_in
        self.conv_in = nn.Conv2d(in_channels = 2 * z_channels if double_z else z_channels, out_channels=block_in, kernel_size=3, padding=1, stride=1)

        self.adaresblocks = nn.ModuleList()

        for i in range(self.n_adaresblock):
            self.adaresblocks.append(AdaINResBlock(in_channels=block_in, style_dim=style_dim))

        self.mid = nn.Module()
        self.mid_block_1 = ResNetBlock(in_channels=block_in, out_channels=block_in, time_embedding_channel=self.time_embedding_channels, dropout=dropout)
        self.mid_attn_1 = LinearAttentionBlock(in_channels=block_in)
        self.mid_block_2 = ResNetBlock(in_channels=block_in, out_channels=block_in, time_embedding_channel=self.time_embedding_channels, dropout=dropout)

        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = intermediate_channels * channel_multipliers[i_level]
            for i_block in range(self.resblock_counts + 1):
                block.append(
                    ResNetBlock(in_channels=block_in, out_channels=block_out, time_embedding_channel=self.time_embedding_channels, dropout=dropout)
                )
                block_in = block_out

                if current_resolution in attn_resolutions:
                    attn.append(
                        LinearAttentionBlock(in_channels=block_out)
                    )
                
            up = nn.Module()
            up.block = block
            up.attn = attn

            if i_level != 0:
                up.upsample = Upsample(block_in, with_conv=True)
                current_resolution = current_resolution * 2
            
            self.up.insert(0, up)
        
        self.norm_out = Normalize(block_in)
        self.conv_out = nn.Conv2d(block_in, out_channels, kernel_size=3, padding=1, stride=1)
    
    def forward(self, z, style, time_embedding=None):
        self.last_z_shape = z.shape

        h = self.conv_in(z)

        for i in range(self.n_adaresblock):
            h = self.adaresblocks[i](h, style)

        h = self.mid_block_1(h, time_embedding)
        h = self.mid_attn_1(h)
        h = self.mid_block_2(h, time_embedding)

        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.resblock_counts + 1):
                h = self.up[i_level].block[i_block](h, time_embedding)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
                    
            if i_level != 0:
                h = self.up[i_level].upsample(h)
        
        if self.give_pre_end:
            return h
        
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        if self.tanh_out:
            h = torch.tanh(h)
        
        return h

class AdaptiveInstanceNorm2D(nn.Module):
    def __init__(self, style_dim, num_features):
        super(AdaptiveInstanceNorm2D, self).__init__()
        self.norm = nn.InstanceNorm2d(num_features)
        self.style_scale_transform = nn.Linear(style_dim, num_features)
        self.style_shift_transform = nn.Linear(style_dim, num_features)
    
    def forward(self, x, style):
        style_scale = self.style_scale_transform(style)[:, :, None, None]
        style_shift = self.style_shift_transform(style)[:, :, None, None]
        out = self.norm(x)
        out = style_scale * out + style_shift
        return out

class AdaINResBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None, style_dim=128):
        super(AdaINResBlock, self).__init__()
        out_channels = out_channels if out_channels is not None else in_channels
        self.block1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, bias=True),
        )
        self.block2 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, bias=True),
        )
        self.adain1 = AdaptiveInstanceNorm2D(style_dim=style_dim, num_features=out_channels)
        self.adain2 = AdaptiveInstanceNorm2D(style_dim=style_dim, num_features=out_channels)

        self.relu = nn.ReLU()
    
    def forward(self, x, style): # ukuran style harusnya (batch_sz, dim, 1, 1) setelah diproses StyleEncoder
        residual = x
        x = self.block1(x)
        x = self.adain1(x, style[:, :, 0, 0])
        x = self.relu(x)
        x = self.block2(x)
        x = self.adain2(x, style[:, :, 0, 0])
        x += residual
        return x
