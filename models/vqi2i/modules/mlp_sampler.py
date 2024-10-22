# This code borrows heavily from Scenimefy by Jiang et. al. (2023)

# File ini dirancang untuk menangani tugas-tugas pemrosesan fitur level rendah seperti ekstraksi patch, normalisasi fitur, dan inisialisasi bobot. 
# Utilitas ini membantu menyiapkan dan mempersiapkan model untuk tugas pembelajaran di mana penanganan patch fitur diperlukan,
#  terutama dalam domain seperti generasi gambar, transfer gaya, atau pembelajaran fitur kontrastif.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

# Inisialisasi bobot jaringan dengan berbagai metode seperti normal, xavier, kaiming, atau orthogonal
def init_weights(net, init_type='normal', init_gain=0.02, debug=False):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    # Inisialisasi bobot jaringan dengan berbagai metode seperti normal, xavier, kaiming, atau orthogonal
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if debug:
                print(classname)
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            # Inisialisasi bias dengan 0
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        # Inisialisasi bobot BatchNorm Layer dengan normal distribution
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    # Apply the initialization function <init_func>
    net.apply(init_func)  # apply the initialization function <init_func>


# Inisialisasi jaringan dengan metode inisialisasi bobot yang berbeda seperti normal, xavier, kaiming, atau orthogonal
def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[], debug=False, initialize_weights=True):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    # Jika gpu_ids tidak kosong, maka gunakan GPU
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        # if not amp:
        # net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs for non-AMP training
    # Jika inisialisasi bobot diaktifkan, maka inisialisasi bobot jaringan
    if initialize_weights:
        init_weights(net, init_type, init_gain=init_gain, debug=debug)
    return net

# Menggunakan normalisasi L2 pada fitur. L2 adalah norm Euclidean yang didefinisikan sebagai akar kuadrat dari jumlah kuadrat elemen tensor.
class Normalize(nn.Module):
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm + 1e-7)
        return out

# PatchSampleF adalah modul yang digunakan untuk mengambil sampel patch dari fitur input. 
class PatchSampleF(nn.Module):
    def __init__(self, use_mlp=False, init_type='normal', init_gain=0.02, nc=256, gpu_ids=[]):
        # potential issues: currently, we use the same patch_ids for multiple images in the batch
        super(PatchSampleF, self).__init__()
        self.l2norm = Normalize(2)
        self.use_mlp = use_mlp
        self.nc = nc  # hard-coded
        self.mlp_init = False
        self.init_type = init_type
        self.init_gain = init_gain
        self.gpu_ids = gpu_ids

    # MLP digunakan untuk mengubah fitur patch yang diambil menjadi representasi yang lebih baik
    def create_mlp(self, feats):
        for mlp_id, feat in enumerate(feats):
            input_nc = feat.shape[1]
            # Sequential digunakan untuk menggabungkan beberapa modul menjadi satu modul
            mlp = nn.Sequential(*[nn.Linear(input_nc, self.nc), nn.ReLU(), nn.Linear(self.nc, self.nc)])
            # Jika gpu_ids tidak kosong, maka gunakan GPU
            if len(self.gpu_ids) > 0:
                mlp.cuda()
            setattr(self, 'mlp_%d' % mlp_id, mlp)
        init_net(self, self.init_type, self.init_gain, self.gpu_ids)
        self.mlp_init = True

    # Fungsi forward untuk mengambil sampel patch dari fitur input
    # num_patches adalah jumlah patch yang diambil dari setiap fitur
    # patch_ids adalah indeks patch yang diambil
    def forward(self, feats, num_patches=64, patch_ids=None):
        return_ids = []
        return_feats = []
        if self.use_mlp and not self.mlp_init:
            self.create_mlp(feats)
        for feat_id, feat in enumerate(feats):
            # B adalah ukuran batch, H adalah tinggi, W adalah lebar
            B, H, W = feat.shape[0], feat.shape[2], feat.shape[3]
            # feat_reshape adalah fitur yang diubah bentuknya. Permute digunakan untuk mengubah urutan dimensi tensor
            feat_reshape = feat.permute(0, 2, 3, 1).flatten(1, 2)
            # Jika num_patches > 0, maka ambil sampel patch dari fitur
            if num_patches > 0:
                if patch_ids is not None:
                    patch_id = patch_ids[feat_id]
                # Jika patch_ids tidak ada, maka ambil sampel patch secara acak
                else:
                    patch_id = torch.randperm(feat_reshape.shape[1], device=feats[0].device)
                    patch_id = patch_id[:int(min(num_patches, patch_id.shape[0]))]  # .to(patch_ids.device)
                x_sample = feat_reshape[:, patch_id, :].flatten(0, 1)  # reshape(-1, x.shape[1])
            # Jika num_patches = 0, maka ambil seluruh fitur
            else:
                x_sample = feat_reshape
                patch_id = []
            # Jika use_mlp = True, maka gunakan MLP. MLP digunakan untuk mengubah fitur patch yang diambil menjadi representasi yang lebih baik
            if self.use_mlp:
                mlp = getattr(self, 'mlp_%d' % feat_id)
                x_sample = mlp(x_sample)
            return_ids.append(patch_id)
            x_sample = self.l2norm(x_sample)

            # Jika num_patches = 0, maka ubah bentuk fitur. Fitur diubah bentuknya menjadi (B, C, H, W)
            if num_patches == 0:
                x_sample = x_sample.permute(0, 2, 1).reshape([B, x_sample.shape[-1], H, W])
            return_feats.append(x_sample)
        return return_feats, return_ids
