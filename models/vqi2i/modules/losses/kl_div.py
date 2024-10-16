import torch
import torch.nn.functional as F

def kl_divergence_loss(s_yr, s_y, s_r, beta=0.2, use_softmax=True):
    if use_softmax:
        s_yr_dist = F.softmax(s_yr, dim=-1)
        s_y_dist = F.softmax(s_y, dim=-1)
        s_r_dist = F.softmax(s_r, dim=-1)
    else:
        s_yr_dist = s_yr
        s_y_dist = s_y
        s_r_dist = s_r

    kl_s_y = F.kl_div(s_yr_dist.log(), s_y_dist, reduction='mean')
    kl_s_r = F.kl_div(s_yr_dist.log(), s_r_dist, reduction='mean')

    return kl_s_y + beta * kl_s_r
