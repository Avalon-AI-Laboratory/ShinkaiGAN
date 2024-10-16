import torch
import torch.nn as nn

class PatchHDCELoss(nn.Module):
    def __init__(self, nce_includes_all_negatives_from_minibatch, batch_sz, nce_temp):
        super(PatchHDCELoss, self).__init__()
        self.nce_includes_all_negatives_from_minibatch = nce_includes_all_negatives_from_minibatch
        self.batch_sz = batch_sz
        self.nce_temp = nce_temp
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')
        self.mask_dtype = torch.bool
    
    def forward(self, feat_q, feat_k, weight=None):
        batch_sz = feat_q.shape[0]
        dim = feat_q.shape[1]
        feat_k = feat_k.detach()

        # Positive Logits
        l_pos = torch.bmm(
            feat_q.view(batch_sz, 1, -1),
            feat_k.view(batch_sz, -1, 1)
        ).view(batch_sz, 1)

        # Negative Logits
        if self.nce_includes_all_negatives_from_minibatch:
            batch_dim_for_bmm = 1
        else:
            batch_dim_for_bmm = self.batch_sz
        
        # Reshape features to batch size
        feat_q = feat_q.view(batch_dim_for_bmm, -1, dim)
        feat_k = feat_k.view(batch_dim_for_bmm, -1, dim)
        n_patches = feat_q.size(1)
        l_neg_curbatch = torch.bmm(feat_q, feat_k.permute(0, 2, 1))

        if weight is not None:
            l_neg_curbatch = l_neg_curbatch * weight
        
        diag = torch.eye(n_patches, dtype=self.mask_dtype, device=feat_q.device)[None, :, :]
        l_neg_curbatch = l_neg_curbatch.masked_fill(diag, -10.0)
        l_neg = l_neg_curbatch.view(-1, n_patches)

        logits = (l_neg - l_pos) / self.nce_temp
        v = torch.logsumexp(logits, dim=1)
        loss_vec = torch.exp(v - v.detach())
        
        out_dummy = torch.cat((l_pos, l_neg), dim=1) / self.nce_temp
        CELoss_dummy = self.ce_loss(out_dummy, torch.zeros(out_dummy.size(0), dtype=torch.long, device=feat_q.device))

        loss = loss_vec.mean() - 1 + CELoss_dummy.detach()

        return loss

def calculate_HDCE_loss(src:torch.Tensor, 
                       tgt:torch.Tensor, 
                       weight,
                       nce_layers:list, 
                       netG:nn.Module, 
                       netF:nn.Module, 
                       n_patch:int = 256, 
                       nce_includes_all_negatives_from_minibatch:bool = False,
                       nce_temp:float = 0.07,
                       lambda_HDCE:float = 0.1, 
                       flip_equivariance:bool = False, 
                       flipped_for_equivariance:bool = False,
                       gen_type: str = "CUT",
                       no_Hneg:str = True):
    n_layers = len(nce_layers)
    criterionHDCE = []

    for i, nce_layer in enumerate(nce_layers):
        criterionHDCE.append(PatchHDCELoss(nce_includes_all_negatives_from_minibatch, src.shape[0], nce_temp))
    with torch.no_grad():
        netG = netG.to("cpu")
        if gen_type == "CUT":
            fake_B_feat = netG(tgt.cpu(), nce_layers, encode_only=True)
        else:
            _, fake_B_feat = netG.content_enc(tgt.cpu(), extract_feats=True, layers_extracted=nce_layers)
    
        if flip_equivariance and flipped_for_equivariance:
            fake_B_feat = [torch.flip(fq, [3]) for fq in fake_B_feat]
    
        if gen_type == "CUT":
            real_A_feat = netG(src.cpu(), nce_layers, encode_only=True)
        else:
            _, real_A_feat = netG.content_enc(src.cpu(), extract_feats=True, layers_extracted=nce_layers)

    for i in range(len(fake_B_feat)):
        fake_B_feat[i] = fake_B_feat[i].to("cuda")
        real_A_feat[i] = real_A_feat[i].to("cuda")
        
    netG = netG.to("cuda")
    feat_q_pool, sample_ids = netF(fake_B_feat, n_patch, None)
    feat_k_pool, _ = netF(real_A_feat, n_patch, sample_ids)

    total_HDCE_loss = 0.0
    for f_q, f_k, crit, nce_layer, w in zip(feat_q_pool, feat_k_pool, criterionHDCE, nce_layers, weight):
        if no_Hneg:
            w = None
        loss = crit(f_q, f_k, w) * lambda_HDCE
        total_HDCE_loss += loss.mean()

    return total_HDCE_loss / n_layers
