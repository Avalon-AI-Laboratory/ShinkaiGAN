import torch
from modules.modular.supervised import calculate_supervised_loss
from modules.modular.unsupervised import calculate_unsupervised_loss
from modules.modular.total_loss import calculate_total_loss

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
    
    
    def train_discriminators(self, x_p, y_p, x, y):
        """Train all discriminator networks"""
        # Train Discriminator Dp1
        self.optimizer_Dp_1.zero_grad()
        s_xp = self.gen.encode_style(x_p, label=1)
        fake_xp, _, _ = self.gen(y_p, label=0, cross=True, s_given=s_xp)
        rec_xp, qloss_xp, _ = self.gen(x_p, label=1, cross=False)
        
        y2x_loss, log = self.gen.loss_a(_, x_p, fake_xp, cond=x_p, optimizer_idx=1, 
                                     last_layer=None, split="train")
        xp_rec_d_loss, _ = self.gen.loss_a(_, x_p, rec_xp, cond=x_p, optimizer_idx=1, 
                                          last_layer=None, split="train")
        
        disc_x_loss = 0.8*y2x_loss + 0.2*xp_rec_d_loss
        disc_x_loss.backward()
        self.optimizer_Dp_1.step()

        # Train Discriminator Dp2
        self.optimizer_Dp_2.zero_grad()
        s_yp = self.gen.encode_style(y_p, label=0)
        s_r = self.gen.encode_style(y, label=-1)
        s_yr = self.gen.style_mix(s_yp, s_r)
        
        fake_yp_mixed, _, _ = self.gen(x_p, label=1, cross=True, s_given=s_yr)
        fake_yp, _, _ = self.gen(x_p, label=1, cross=True, s_given=s_yp)
        rec_yp, qloss_yp, _ = self.gen(y_p, label=0, cross=False)
        
        x2y_loss, log = self.gen.loss_b(_, y_p, fake_yp_mixed, cond=y_p, 
                                     optimizer_idx=1, last_layer=None, split="train")
        yp_rec_d_loss, _ = self.gen.loss_b(_, y_p, rec_yp, cond=y_p, 
                                          optimizer_idx=1, last_layer=None, split="train")
        
        disc_y_loss = 0.8*x2y_loss + 0.2*yp_rec_d_loss
        disc_y_loss.backward()
        self.optimizer_Dp_2.step()

        # Train Discriminator Du
        self.optimizer_Du.zero_grad()
        fake_y, _, _ = self.gen(x, label=1, cross=True, s_given=s_r)
        rec_y, qloss_y, _ = self.gen(y, label=1, cross=True, s_given=s_r)
        
        x2r_loss, log = self.gen.loss_c(_, y, fake_y, optimizer_idx=1, 
                                     last_layer=None, split="train")
        y_rec_d_loss, _ = self.gen.loss_c(_, y, rec_y, optimizer_idx=1, 
                                         last_layer=None, split="train")
        
        disc_r_loss = 0.8*x2r_loss + 0.2*y_rec_d_loss
        disc_r_loss.backward()
        self.optimizer_Du.step()

        return fake_xp, fake_yp, fake_yp_mixed, fake_y, rec_xp, rec_yp, rec_y, \
               qloss_xp, qloss_yp, qloss_y, s_xp, s_yp, s_yr, s_r

    def train(self):
        for epoch in range(self.epoch_start, self.epoch_end):
            lambda_sup = torch.cos(torch.tensor((torch.pi*(epoch - 1)/(self.epoch_end*2)))).to(self.device)
            
            for i in range(self.iterations):
                torch.cuda.synchronize()
                self.gen.train()
                self.F.train()

                # Get data
                x_p, y_p, x, y = next(iter(self.train_loader)).values()
                x_p, y_p, x, y = x_p.to(self.device), y_p.to(self.device), \
                                x.to(self.device), y.to(self.device)
                
                # Train discriminators and get generated images
                fake_xp, fake_yp, fake_yp_mixed, fake_y, rec_xp, rec_yp, rec_y, \
                qloss_xp, qloss_yp, qloss_y, s_xp, s_yp, s_yr, s_r = \
                    self.train_discriminators(x_p, y_p, x, y)

                # Train Autoencoder
                self.optimizer_AE.zero_grad()

                # Calculate supervised and unsupervised losses
                l_supervised = calculate_supervised_loss(
                    self.gen, self.F, x_p, y_p, fake_xp, fake_yp, fake_yp_mixed,
                    rec_xp, rec_yp, s_xp, s_yp, s_yr, qloss_xp, qloss_yp
                )

                l_unsupervised = calculate_unsupervised_loss(
                    self.gen, self.F, x, y, fake_y, rec_y, qloss_y,
                    self.n_patches, epoch
                )

                # Calculate total loss
                loss = calculate_total_loss(l_supervised, l_unsupervised, lambda_sup)

                loss.backward()
                self.optimizer_AE.step()
                
                if self.optimizer_F is None:
                    self.optimizer_F = torch.optim.Adam(
                        self.F.parameters(),
                        lr=1e-4, betas=(0.5, 0.999)
                    )
                self.optimizer_F.step()
                
                torch.cuda.empty_cache()
                if i % 1 == 0:
                    print(f"Epoch [{epoch}/{self.epoch_end}], "
                          f"Iteration [{i}/{self.iterations}], "
                          f"Loss: {loss.item()}")
                    print(f"Loss supervised: {l_supervised.item()}, "
                          f"Loss unsupervised: {l_unsupervised.item()}")