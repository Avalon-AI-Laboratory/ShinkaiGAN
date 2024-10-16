import os
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.vqi2i.modules.losses.patchNCE import calculate_NCE_loss
from models.vqi2i.modules.losses.SRC import calculate_R_loss
from models.vqi2i.modules.losses.patchHDCE import calculate_HDCE_loss
from models.vqi2i.modules.losses.kl_div import kl_divergence_loss

from models.CUT.resnetG import ResnetGenerator
from models.vqi2i.modules.mlp_sampler import PatchSampleF
from models.vqi2i.vqgan_model.vqi2i_adain import VQI2ICrossGAN_AdaIN

from data_modules.datasetClass import myDataset
from torch.utils.data import DataLoader

class Trainer:
    def __init__(self,
                 gen,
                 F,
                 save_dir,
                 device,
                 train_loader,
                 val_loader,
                 scheduler,
                 epoch_start,
                 epoch_end,
                 n_patches=256,
                 iterations=60000):
        self.gen = gen
        self.F = F
        self.save_dir = save_dir
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer_AE = torch.optim.Adam(
            list(self.gen.content_enc.parameters()) + \
            list(self.gen.style_enc_a.parameters()) + \
            list(self.gen.style_enc_b.parameters()) + \
            list(self.gen.shinkai_style_enc.parameters()) + \
            list(self.gen.quantize.parameters()) + \
            list(self.gen.quant_conv.parameters()) + \
            list(self.gen.post_quant_conv.parameters()) + \
            list(self.gen.decoder_a.parameters()) + \
            list(self.gen.decoder_b.parameters()) + \
            list(self.gen.style_mixer_mlp.parameters()),
            lr=1e-4, betas=((0.5, 0.999))
        )

        self.optimizer_Dp_1 = torch.optim.Adam(
            self.gen.loss_a.discriminator.parameters(),
            lr=1e-4, betas=((0.5, 0.999))
        )
        self.optimizer_Dp_2 = torch.optim.Adam(
            self.gen.loss_b.discriminator.parameters(),
            lr=1e-4, betas=((0.5, 0.999))
        )
        self.optimizer_Du = torch.optim.Adam(
            self.gen.loss_c.discriminator.parameters(),
            lr=1e-4, betas=((0.5, 0.999))
        )
        try:
            self.optimizer_F = torch.optim.Adam(
                self.F.parameters(),
                lr=1e-4, betas=((0.5, 0.999))
            )
        except:
            self.optimizer_F = None

        self.scheduler = scheduler
        self.epoch_start = epoch_start
        self.epoch_end = epoch_end
        self.n_patches = n_patches
        self.iterations = iterations
    
    def train(self):
        for epoch in range(self.epoch_start, self.epoch_end):
            lambda_sup = torch.cos(torch.tensor((torch.pi*(epoch - 1)/(self.epoch_end*2)))).to(self.device)
            for i in range(self.iterations):
                self.gen.train()
                self.F.train()
                self.optimizer_AE.zero_grad()
                self.optimizer_Dp_2.zero_grad()
                self.optimizer_Du.zero_grad()

                # Dapatkan data
                x_p, y_p, x, y = next(iter(self.train_loader)).values()
                x_p, y_p, x, y = x_p.to(self.device), y_p.to(self.device), x.to(self.device), y.to(self.device)
                # x_p, y_p, x, y = x_p.unsqueeze(0), y_p.unsqueeze(0), x.unsqueeze(0), y.unsqueeze(0)
                
                # ========== SUPERVISED TRAINING BRANCH ==========
                # Latih Diskriminator Dp1
                self.optimizer_Dp_1.zero_grad()

                s_xp = self.gen.encode_style(x_p, label=1)
                fake_xp, _, _ = self.gen(y_p, label=0, cross=True, s_given=s_xp)
                rec_xp, qloss_xp, _ = self.gen(x_p, label=1, cross=False)

                y2x_loss, log = self.gen.loss_a(_, x_p, fake_xp, cond=x_p, optimizer_idx=1, last_layer=None, split="train")
                xp_rec_d_loss, _ = self.gen.loss_a(_, x_p, rec_xp, cond=x_p, optimizer_idx=1, last_layer=None, split="train")

                disc_x_loss = 0.8*y2x_loss + 0.2*xp_rec_d_loss
                disc_x_loss.backward()
                self.optimizer_Dp_1.step()

                # Latih Diskriminator Dp2
                self.optimizer_Dp_2.zero_grad()

                s_yp = self.gen.encode_style(y_p, label=0)
                s_r = self.gen.encode_style(y, label=-1)
                s_yr = self.gen.style_mix(s_yp, s_r)

                fake_yp_mixed, _, _ = self.gen(x_p, label=1, cross=True, s_given=s_yr)
                fake_yp, _, _ = self.gen(x_p, label=1, cross=True, s_given=s_yp)
                rec_yp, qloss_yp, _ = self.gen(y_p, label=0, cross=False)

                x2y_loss, log = self.gen.loss_b(_, y_p, fake_yp_mixed, cond=y_p, optimizer_idx=1, last_layer=None, split="train")
                yp_rec_d_loss, _ = self.gen.loss_b(_, y_p, rec_yp, cond=y_p, optimizer_idx=1, last_layer=None, split="train")
                # yp_independent_d_loss, _ = self.gen.loss_b(_, y_p, fake_yp, cond=y_p, optimizer_idx=1, last_layer=None, split="train")

                disc_y_loss = 0.8*x2y_loss + 0.2*yp_rec_d_loss
                # disc_y_loss = 0.8*x2y_loss + 0.1*yp_rec_d_loss + 0.1*yp_independent_d_loss # Alternatif

                disc_y_loss.backward()
                self.optimizer_Dp_2.step()

                # Latih Diskriminator Du
                self.optimizer_Du.zero_grad()

                fake_y, _, _ = self.gen(x, label=1, cross=True, s_given=s_r)
                rec_y, qloss_y, _ = self.gen(y, label=1, cross=True, s_given=s_r)

                x2r_loss, log = self.gen.loss_c(_, y, fake_y, optimizer_idx=1, last_layer=None, split="train")
                y_rec_d_loss, _ = self.gen.loss_c(_, y, rec_y, optimizer_idx=1, last_layer=None, split="train")

                disc_r_loss = 0.8*x2r_loss + 0.2*y_rec_d_loss
                
                disc_r_loss.backward()
                self.optimizer_Du.step()

                # Latih Autoencoder
                self.optimizer_AE.zero_grad()

                ae_lossA, _ = self.gen.loss_a(qloss_xp, x_p, rec_xp, fake=fake_xp, cond=x_p, switch_weight=0.1, optimizer_idx=0, last_layer=self.gen.get_last_layer(label=1), split="train")
                ae_lossB, _ = self.gen.loss_b(qloss_yp, y_p, rec_yp, fake=fake_yp_mixed, cond=y_p, switch_weight=0.1, optimizer_idx=0, last_layer=self.gen.get_last_layer(label=0), split="train")
                ae_lossC, _ = self.gen.loss_c(qloss_y, y, rec_y, fake=fake_y, switch_weight=0.1, optimizer_idx=0, last_layer=self.gen.get_last_layer(label=0), split="train")

                AtoBtoA, _, s_xp_fake = self.gen(fake_xp, label=1, cross=False)
                BtoAtoB, _, s_y_fake = self.gen(fake_yp, label=0, cross=False)
                BtoAtoBR, _, s_y_r_fake = self.gen(fake_yp_mixed, label=0, cross=False)

                style_loss_xp = torch.mean(torch.abs(s_xp.detach() - s_xp_fake)).to(self.device)
                style_loss_yp = torch.mean(torch.abs(s_yp.detach() - s_y_fake)).to(self.device) # bagian ini perlu diperbaiki
                style_loss_xyr = torch.mean(torch.abs(s_yr.detach() - s_y_r_fake)).to(self.device) # bagian ini perlu juga diperbaiki
                style_loss  = 0.3 * (style_loss_xp + style_loss_yp + style_loss_xyr)

                style_mix_loss = 20*kl_divergence_loss(s_yr.squeeze(-2, -1), s_yp.squeeze(-2, -1), s_r.squeeze(-2, -1), use_softmax=True)

                c_xp, c_xp_quantized = self.gen.encode_content(x_p)
                c_yp, c_yp_quantized = self.gen.encode_content(fake_yp) # salah
                content_loss = torch.mean(torch.abs(c_xp.detach() - c_yp)).to(self.device)
                content_quantized_loss = torch.mean(torch.abs(c_xp_quantized.detach() - c_yp_quantized)).to(self.device)  
                content_loss = 0.5 * (content_loss + content_quantized_loss)

                pixel_anime_idt = torch.mean(torch.abs(y.detach() - rec_y))
                s_r_rec = self.gen.encode_style(rec_y, label=-1)
                style_anime_idt = torch.mean(torch.abs(s_r.detach() - s_r_rec))
                c_y_rec, c_y_rec_q = self.gen.encode_content(rec_y)
                c_y, c_y_q = self.gen.encode_content(y)
                content_anime_idt = torch.mean(torch.abs(c_y_rec.detach() - c_y)) + torch.mean(torch.abs(c_y_rec_q.detach() - c_y_q))
                idt_loss = 0.3 * (pixel_anime_idt + style_anime_idt + 0.5*content_anime_idt)

                cross_recons_loss_a = torch.mean(torch.abs(x_p.detach() - fake_xp)).to("cuda")
                cross_recons_loss_b = torch.mean(torch.abs(y_p.detach() - fake_yp)).to("cuda")
                cross_recons_loss = 0.5 * (cross_recons_loss_a + cross_recons_loss_b)

                nce_loss = calculate_NCE_loss(y_p, fake_yp, [0,4,9,12,14], self.gen, self.F, gen_type="ShinkaiGAN")
                
                l_supervised = ae_lossA + ae_lossB + 0.5*(style_loss + content_loss) + 0.001*cross_recons_loss + style_mix_loss + 0.3*nce_loss

                # ========== UNSUPERVISED TRAINING BRANCH ==========
                c_y_fake, c_y_fake_q = self.gen.encode_content(fake_y)
                c_x, c_x_quantized = self.gen.encode_content(x)
                L_cP = torch.mean(torch.abs(c_y_fake.detach() - c_x)).to(self.device)
                L_cP_q = torch.mean(torch.abs(c_y_fake_q.detach() - c_x_quantized)).to(self.device)
                L_cP = 0.5 * (L_cP + L_cP_q)

                s_x_fake = self.gen.encode_style(fake_y, label=1)
                s_x = self.gen.encode_style(x, label=1)
                L_sP = torch.mean(torch.abs(s_x_fake.detach() - s_x)).to(self.device)

                l_src, weight = calculate_R_loss(x, fake_y, [0,4,9,12,14], self.gen, self.F, self.n_patches, 0.05, epoch=epoch, gen_type="ShinkaiGAN")

                l_hdce = calculate_HDCE_loss(x, fake_y, weight, [0,4,9,12,14], self.gen, self.F, gen_type="ShinkaiGAN")

                l_unsupervised = ae_lossC + L_cP + L_sP + 0.5*(l_src + l_hdce) + 0.1*idt_loss

                # ========== TOTAL LOSS ==========
                loss = l_unsupervised + lambda_sup * l_supervised

                loss.backward()
                self.optimizer_AE.step()
                if self.optimizer_F == None:
                    self.optimizer_F = torch.optim.Adam(
                        self.F.parameters(),
                        lr=1e-4, betas=((0.5, 0.999))
                    )
                self.optimizer_F.step()

                if i % 1 == 0:
                    print(f"Epoch [{epoch}/{self.epoch_end}], Iteration [{i}/{self.iterations}], Loss: {loss.item()}")
                    print(f"Loss supervised: {l_supervised.item()}, Loss unsupervised: {l_unsupervised.item()}")
