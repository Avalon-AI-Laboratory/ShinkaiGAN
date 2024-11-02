# supervised_loss.py
import torch
from models.vqi2i.modules.losses.patchNCE import calculate_NCE_loss
from models.vqi2i.modules.losses.kl_div import kl_divergence_loss

def calculate_supervised_loss(gen, F, x_p, y_p, fake_xp, fake_yp, fake_yp_mixed, rec_xp, rec_yp, s_xp, s_yp, s_yr, s_r, qloss_xp, qloss_yp):
    # Calculate AE losses
    ae_lossA, _ = gen.loss_a(qloss_xp, x_p, rec_xp, fake=fake_xp, cond=x_p, 
                            switch_weight=0.1, optimizer_idx=0, 
                            last_layer=gen.get_last_layer(label=1), split="train")
    
    ae_lossB, _ = gen.loss_b(qloss_yp, y_p, rec_yp, fake=fake_yp_mixed, cond=y_p, 
                            switch_weight=0.1, optimizer_idx=0, 
                            last_layer=gen.get_last_layer(label=0), split="train")
    

    # Calculate cycle reconstructions
    AtoBtoA, _, s_xp_fake = gen(fake_xp, label=1, cross=False)
    BtoAtoB, _, s_y_fake = gen(fake_yp, label=0, cross=False)
    BtoAtoBR, _, s_y_r_fake = gen(fake_yp_mixed, label=0, cross=False)

    # Style losses
    style_loss_xp = torch.mean(torch.abs(s_xp.detach() - s_xp_fake))
    style_loss_yp = torch.mean(torch.abs(s_yp.detach() - s_y_fake)) # bagian ini perlu diperbaiki
    style_loss_xyr = torch.mean(torch.abs(s_yr.detach() - s_y_r_fake)) # bagian ini perlu juga diperbaiki
    style_loss = 0.3 * (style_loss_xp + style_loss_yp + style_loss_xyr)

    # Style mixing loss
    style_mix_loss = 20 * kl_divergence_loss(s_yr.squeeze(-2, -1), 
                                            s_yp.squeeze(-2, -1), 
                                            s_r.squeeze(-2, -1), 
                                            use_softmax=True)

    # Content losses
    c_xp, c_xp_quantized = gen.encode_content(x_p)
    c_yp, c_yp_quantized = gen.encode_content(fake_yp) #salah
    content_loss = torch.mean(torch.abs(c_xp.detach() - c_yp))
    content_quantized_loss = torch.mean(torch.abs(c_xp_quantized.detach() - c_yp_quantized))
    content_loss = 0.5 * (content_loss + content_quantized_loss)

    # Cross reconstruction losses
    cross_recons_loss_a = torch.mean(torch.abs(x_p.detach() - fake_xp))
    cross_recons_loss_b = torch.mean(torch.abs(y_p.detach() - fake_yp))
    cross_recons_loss = 0.5 * (cross_recons_loss_a + cross_recons_loss_b)

    # NCE loss
    nce_loss = calculate_NCE_loss(y_p, fake_yp, [0,4,9,12], gen, F, gen_type="ShinkaiGAN")

    # Total supervised loss
    total_loss = (ae_lossA + ae_lossB + 
                 0.5 * (style_loss + content_loss) + 
                 0.001 * cross_recons_loss + 
                 style_mix_loss + 
                 0.3 * nce_loss)

    return total_loss

