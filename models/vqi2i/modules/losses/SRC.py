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
        self.n_epochs = n_epochs if n_epochs is not None else 10
        self.n_epochs_decay = n_epochs_decay if n_epochs_decay is not None else 10
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

        spatial_q = torch.bmm(feat_q_v, feat_q_v.transpose(2, 1))
        spatial_k = torch.bmm(feat_k_v, feat_k_v.transpose(2, 1))

        weight_seed = spatial_k.clone().detach()
        diag = torch.eye(weight_seed.shape[1], dtype=self.mask_dtype, device=feat_k_v.device)[None, :, :]
        # diag = torch.eye(self.n_patches, dtype=self.mask_dtype, device=feat_k_v.device)[None, :, :]

        HDCE_gamma = self.HDCE_gamma
        if self.use_curriculum:
            HDCE_gamma = HDCE_gamma + (self.HDCE_gamma_min - self.HDCE_gamma) * (epoch) / (self.n_epochs + self.n_epochs_decay)
            if self.step_gamma & (epoch > self.step_gamma_epoch):
                HDCE_gamma = 1
        
        # Weights by semantic relation
        weight_seed.masked_fill_(diag, -10.0)
        weight_out = nn.Softmax(dim=2)(weight_seed.clone() / HDCE_gamma).detach()
        wmax_out, _ = torch.max(weight_out, 2, keepdim=True)
        weight_out /= wmax_out

        if only_weight:
            return 0, weight_out

        spatial_q = nn.Softmax(dim=1)(spatial_q)
        spatial_k = nn.Softmax(dim=1)(spatial_k).detach()

        loss_SRC = self.get_jsd(spatial_q, spatial_k)

        return loss_SRC, weight_out

    def get_jsd(self, p1, p2):
        m = 0.5 * (p1 + p2)
        jsd = 0.5 * (nn.KLDivLoss(reduction='sum', log_target=True)(torch.log(m), torch.log(p1))
                     + nn.KLDivLoss(reduction='sum', log_target=True)(torch.log(m), torch.log(p2)))
        return jsd

def calculate_R_loss(src:torch.Tensor, 
                     tgt:torch.Tensor, 
                     nce_layers:list,
                     net_G:nn.Module,
                     netF:nn.Module,
                     n_patch:int,
                     lambda_SRC:float,
                     flip_equivariance:bool = False, 
                     flipped_for_equivariance:bool = False,
                     gen_type: str = "CUT",
                     only_weight=False, 
                     epoch=None,
                     n_epochs=10,
                     n_epochs_decay=10):
    n_layers = len(nce_layers)
    criterionSRC = []

    for i, nce_layer in enumerate(nce_layers):
        criterionSRC.append(SRC_Loss(n_patches=n_patch, 
                                     HDCE_gamma=50, 
                                     use_curriculum=True, 
                                     HDCE_gamma_min=10, 
                                     n_epochs=n_epochs, 
                                     n_epochs_decay=n_epochs_decay,
                                     step_gamma=True,
                                     step_gamma_epoch=200).to("cuda"))
    with torch.no_grad():
        net_G = net_G.to("cpu")
        if gen_type == "CUT":
            fake_B_feat = net_G(tgt.cpu(), nce_layers, encode_only=True)
        else:
            _, fake_B_feat = net_G.content_enc(tgt.cpu(), extract_feats=True, layers_extracted=nce_layers)
    
        if flip_equivariance and flipped_for_equivariance:
            fake_B_feat = [torch.flip(fq, [3]) for fq in fake_B_feat]
    
        if gen_type == "CUT":
            real_A_feat = net_G(src.cpu(), nce_layers, encode_only=True)
        else:
            _, real_A_feat = net_G.content_enc(src.cpu(), extract_feats=True, layers_extracted=nce_layers)

    for i in range(len(fake_B_feat)):
        fake_B_feat[i] = fake_B_feat[i].to("cuda")
        real_A_feat[i] = real_A_feat[i].to("cuda")
    net_G = net_G.to("cuda")
    fake_B_pool, sample_ids = netF(fake_B_feat, n_patch, None)
    real_A_pool, _ = netF(real_A_feat, n_patch, sample_ids)

    total_SRC_loss = 0.0
    weights = []

    for f_q, f_k, crit, nce_layer in zip(fake_B_pool, real_A_pool, criterionSRC, nce_layers):
        loss_SRC, weight = crit(f_q, f_k, only_weight, epoch)
        total_SRC_loss += loss_SRC * lambda_SRC
        weights.append(weight)
    return total_SRC_loss / n_layers, weights
