# unsupervised_loss.py
import torch

from models.vqi2i.modules.losses.SRC import calculate_R_loss
from models.vqi2i.modules.losses.patchHDCE import calculate_HDCE_loss

def calculate_unsupervised_loss(gen, F, x, y, fake_y, rec_y, qloss_y, n_patches, epoch):
    # Content preservation losses
    c_y_fake, c_y_fake_q = gen.encode_content(fake_y)
    c_x, c_x_quantized = gen.encode_content(x)
    L_cP = torch.mean(torch.abs(c_y_fake.detach() - c_x))
    L_cP_q = torch.mean(torch.abs(c_y_fake_q.detach() - c_x_quantized))
    L_cP = 0.5 * (L_cP + L_cP_q)

    # Style preservation losses
    s_x_fake = gen.encode_style(fake_y, label=1)
    s_x = gen.encode_style(x, label=1)
    L_sP = torch.mean(torch.abs(s_x_fake.detach() - s_x))

    # Identity losses
    pixel_anime_idt = torch.mean(torch.abs(y.detach() - rec_y))
    s_r_rec = gen.encode_style(rec_y, label=-1)
    s_r = gen.encode_style(y, label=-1)
    style_anime_idt = torch.mean(torch.abs(s_r.detach() - s_r_rec))
    
    c_y_rec, c_y_rec_q = gen.encode_content(rec_y)
    c_y, c_y_q = gen.encode_content(y)
    content_anime_idt = (torch.mean(torch.abs(c_y_rec.detach() - c_y)) + 
                        torch.mean(torch.abs(c_y_rec_q.detach() - c_y_q)))
    idt_loss = 0.3 * (pixel_anime_idt + style_anime_idt + 0.5 * content_anime_idt)

    # Calculate SRC loss
    l_src, weight = calculate_R_loss(x, fake_y, [0,4,9,12], gen, F, n_patches, 
                                   0.05, epoch=epoch, gen_type="ShinkaiGAN")
    for item in weight:
        item = item.detach()

    # Calculate HDCE loss
    l_hdce = calculate_HDCE_loss(x, fake_y, weight, [0,4,9,12], gen, F, 
                                gen_type="ShinkaiGAN")

    # Calculate AE loss
    ae_lossC, _ = gen.loss_c(qloss_y, y, rec_y, fake=fake_y, 
                            switch_weight=0.1, optimizer_idx=0, 
                            last_layer=gen.get_last_layer(label=0), split="train")

    # Total unsupervised loss
    total_loss = (ae_lossC + L_cP + L_sP + 
                 0.5 * (l_src + l_hdce) + 
                 0.1 * idt_loss)

    return total_loss
