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
        l_neg_curbatch = torch.bmm(feat_q, feat_k.permute(0, 2, 1))

        if weight is not None:
            l_neg_curbatch = l_neg_curbatch * weight
        
        diag = torch.eye(n_patches, dtype=self.mask_dtype, device=feat_q.device)[None, :, :]
        l_neg_curbatch = l_neg_curbatch.masked_fill(diag, -10.0)
        l_neg = l_neg_curbatch.view(-1, n_patches)

        out = torch.cat((l_pos, l_neg), dim=1) / self.nce_temp

        loss = self.ce_loss(out, torch.zeros(out.size(0), dtype=torch.long, device=feat_q.device))

        return loss
