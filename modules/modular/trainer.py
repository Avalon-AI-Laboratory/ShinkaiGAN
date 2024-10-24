import torch
import os
from modules.modular.supervised import calculate_supervised_loss
from modules.modular.unsupervised import calculate_unsupervised_loss
from modules.modular.total_loss import calculate_total_loss
from modules.modular.validation import Validator
from modules.modular.testing import Tester

class Trainer:
    def __init__(self,
                 gen,
                 F,
                 save_dir,
                 device,
                 train_loader,
                 val_loader,
                 test_loader,  # Added test_loader
                 scheduler,
                 epoch_start,
                 epoch_end,
                 n_patches=256,
                 iterations=60000,
                 test_frequency=10):  # Added test_frequency parameter
        self.gen = gen
        self.F = F
        self.save_dir = save_dir
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader  # Store test_loader
        self.scheduler = scheduler
        self.epoch_start = epoch_start
        self.epoch_end = epoch_end
        self.n_patches = n_patches
        self.iterations = iterations
        self.test_frequency = test_frequency

        # Initialize optimizers
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

        # Initialize validator and tester
        self.validator = Validator(gen, F, device, val_loader, save_dir)
        self.tester = Tester(gen, F, device, test_loader, save_dir)

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

    def save_checkpoint(self, epoch, iteration, metrics=None):
        """Save model checkpoint with additional testing metrics"""
        checkpoint_path = os.path.join(self.save_dir, f'checkpoint_epoch_{epoch}.pth')
        checkpoint = {
            'epoch': epoch,
            'iteration': iteration,
            'generator_state_dict': self.gen.state_dict(),
            'feature_extractor_state_dict': self.F.state_dict(),
            'optimizer_AE_state_dict': self.optimizer_AE.state_dict(),
            'optimizer_F_state_dict': self.optimizer_F.state_dict() if self.optimizer_F else None,
            'optimizer_Dp_1_state_dict': self.optimizer_Dp_1.state_dict(),
            'optimizer_Dp_2_state_dict': self.optimizer_Dp_2.state_dict(),
            'optimizer_Du_state_dict': self.optimizer_Du.state_dict(),
        }
        
        # Include test metrics if available
        if metrics is not None:
            checkpoint['test_metrics'] = metrics
            
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.gen.load_state_dict(checkpoint['generator_state_dict'])
        self.F.load_state_dict(checkpoint['feature_extractor_state_dict'])
        self.optimizer_AE.load_state_dict(checkpoint['optimizer_AE_state_dict'])
        if checkpoint['optimizer_F_state_dict'] is not None:
            self.optimizer_F.load_state_dict(checkpoint['optimizer_F_state_dict'])
        self.optimizer_Dp_1.load_state_dict(checkpoint['optimizer_Dp_1_state_dict'])
        self.optimizer_Dp_2.load_state_dict(checkpoint['optimizer_Dp_2_state_dict'])
        self.optimizer_Du.load_state_dict(checkpoint['optimizer_Du_state_dict'])
        
        return checkpoint['epoch'], checkpoint['iteration']

    def train(self):
        """Main training loop with integrated testing"""
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
                
                # Validate every 30 iterations
                if i % 30 == 0:
                    self.validator.validate(
                        epoch, i,
                        save_model=False,
                        optimizer_AE=self.optimizer_AE,
                        optimizer_F=self.optimizer_F,
                        optimizer_Dp_1=self.optimizer_Dp_1,
                        optimizer_Dp_2=self.optimizer_Dp_2,
                        optimizer_Du=self.optimizer_Du
                    )
                
                torch.cuda.empty_cache()
                if i % 1 == 0:
                    print(f"Epoch [{epoch}/{self.epoch_end}], "
                          f"Iteration [{i}/{self.iterations}], "
                          f"Loss: {loss.item()}")
                    print(f"Loss supervised: {l_supervised.item()}, "
                          f"Loss unsupervised: {l_unsupervised.item()}")
            
            # Run testing and save checkpoint every test_frequency epochs
            if (epoch + 1) % self.test_frequency == 0:
                print(f"\nRunning test evaluation at epoch {epoch + 1}...")
                test_metrics = self.tester.test()
                
                # Save checkpoint with test metrics
                self.save_checkpoint(epoch, i, test_metrics)
                
                # Print test metrics
                print("\nTest Metrics:")
                for domain, metrics in test_metrics.items():
                    print(f"\n{domain}:")
                    for metric, value in metrics.items():
                        print(f"{metric}: {value:.4f}")
            
            # Save regular checkpoint for other epochs
            elif (epoch + 1) % 10 == 0:
                self.save_checkpoint(epoch, i)

    def evaluate(self, checkpoint_path=None):
        """Run evaluation on test set"""
        if checkpoint_path is not None:
            self.load_checkpoint(checkpoint_path)
        
        print("Running full evaluation on test set...")
        test_metrics = self.tester.test()
        
        return test_metrics