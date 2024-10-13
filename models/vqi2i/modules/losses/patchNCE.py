import torch
import torch.nn as nn

class PatchNCELoss(nn.Module):
    def __init__(self, nce_includes_all_negatives_from_minibatch, batch_sz, nce_temp):
        super(PatchNCELoss, self).__init__()
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
        l_neg_curbatch = torch.bmm(feat_q, feat_k.transpose(2, 1))

        if weight is not None:
            l_neg_curbatch *= weight
        
        diag = torch.eye(n_patches, dtype=self.mask_dtype, device=feat_q.device)[None, :, :]
        l_neg_curbatch = l_neg_curbatch.masked_fill(diag, -10.0)
        l_neg = l_neg_curbatch.view(-1, n_patches)

        out = torch.cat((l_pos, l_neg), dim=1) / self.nce_temp

        loss = self.ce_loss(out, torch.zeros(out.size(0), dtype=torch.long, device=feat_q.device))

        return loss


def calculate_NCE_loss(src:torch.Tensor, 
                       tgt:torch.Tensor, 
                       nce_layers:list, 
                       netG:nn.Module, 
                       netF_s:nn.Module, 
                       n_patch:int = 256, 
                       nce_includes_all_negatives_from_minibatch:bool = False,
                       nce_temp:float = 0.07,
                       lambda_NCE_s:float = 0.1, 
                       flip_equivariance:bool = False, 
                       flipped_for_equivariance:bool = False,
                       gen_type: str = "CUT"):
    """
    The code borrows heavily from the PyTorch implementation of CUT
    https://github.com/taesungp/contrastive-unpaired-translation
    """
    n_layers = len(nce_layers)
    criterionNCE = []

    for i, nce_layer in enumerate(nce_layers):
        criterionNCE.append(PatchNCELoss(nce_includes_all_negatives_from_minibatch, src.shape[0], nce_temp).cuda())

    if gen_type == "CUT":
        feat_q = netG(tgt, nce_layers, encode_only=True)
    else:
        _, feat_q = netG.content_enc(tgt, extract_feats=True, layers_extracted=nce_layers)

    if flip_equivariance and flipped_for_equivariance:
        feat_q = [torch.flip(fq, [3]) for fq in feat_q]

    if gen_type == "CUT":
        feat_k = netG(src, nce_layers, encode_only=True)
    else:
        _, feat_k = netG.content_enc(src, extract_feats=True, layers_extracted=nce_layers)
    
    feat_k_pool, sample_ids = netF_s(feat_k, n_patch, None)
    feat_q_pool, _ = netF_s(feat_q, n_patch, sample_ids)

    total_nce_loss = 0.0
    for f_q, f_k, crit, nce_layer in zip(feat_q_pool, feat_k_pool, criterionNCE, nce_layers):
        loss = crit(f_q, f_k) * lambda_NCE_s
        total_nce_loss += loss.mean()

    return total_nce_loss / n_layers
