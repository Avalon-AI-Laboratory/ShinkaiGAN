import torch
import torch.nn as nn

class Normalize(nn.Module):
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power
    
    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1./self.power)
        out = x.div(norm + 1e-7)
        return out

class SRC_Loss(nn.Module):
    def __init__(self, n_patches, HDCE_gamma, use_curriculum, HDCE_gamma_min, n_epochs, n_epochs_decay, step_gamma, step_gamma_epoch):
        super(SRC_Loss, self).__init__()
        self.n_patches = n_patches
        self.HDCE_gamma = HDCE_gamma
        self.use_curriculum = use_curriculum
        self.HDCE_gamma_min = HDCE_gamma_min
        self.n_epochs = n_epochs
        self.n_epochs_decay = n_epochs_decay
        self.mask_dtype = torch.bool
        self.step_gamma = step_gamma
        self.step_gamma_epoch = step_gamma_epoch
    
    def forward(self, feat_q, feat_k, only_weight=False, epoch=None):
        batch_sz = feat_q.shape[0]
        dim = feat_q.shape[1]
        feat_k = feat_k.detach()
        batch_dim_for_bmm = 1
        feat_k = Normalize()(feat_k)
        feat_q = Normalize()(feat_q)

        # SRC
        feat_q_v = feat_q.view(batch_dim_for_bmm, -1, dim)
        feat_k_v = feat_k.view(batch_dim_for_bmm, -1, dim)

        spatial_q = torch.bmm(feat_q_v, feat_k_v.permute(0, 2, 1))
        spatial_k = torch.bmm(feat_k_v, feat_q_v.permute(0, 2, 1))

        weight_seed = spatial_k.clone().detach()
        diag = torch.eye(self.n_patches, dtype=self.mask_dtype, device=feat_q.device)[None, :, :]

        HDCE_gamma = self.HDCE_gamma
        if self.use_curriculum:
            HDCE_gamma = HDCE_gamma + (self.HDCE_gamma_min - self.HDCE_gamma) * (epoch) / (self.n_epochs + self.n_epochs_decay)
            if self.step_gamma & (epoch > self.step_gamma_epoch):
                HDCE_gamma = 1
        
        # Weights by semantic relation
        weight_seed.masked_fill_(diag, -10.0)
        weight_out = nn.Softmax(dim=2)(weight_seed.clone() / HDCE_gamma).detach()
        wmax_out, _ = torch.max(weight_out, 2, keepdim=True)
        weight_out = weight_out / wmax_out

        if only_weight:
            return 0, weight_out

        spatial_q = nn.Softmax(dim=1)(spatial_q)
        spatial_k = nn.Softmax(dim=1)(spatial_k).detach()

        loss_SRC = self.get_jsd(spatial_q, spatial_k)

        return loss_SRC, weight_out

    def get_jsd(self, p1, p2):
        m = 0.5 * (p1 + p2)
        jsd = 0.5 * (nn.KLDivLoss(reduction='sum', log_target=True)(torch.log(m), torch.log(p1)),
                     + nn.KLDivLoss(reduction='sum', log_target=True)(torch.log(m), torch.log(p2)))
        return jsd
