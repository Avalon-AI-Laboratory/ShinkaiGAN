import torch
import os
from datetime import datetime
from torchvision.utils import save_image
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
                 val_loader=None,
                 test_loader=None,
                 scheduler=None,
                 epoch_start=1,
                 epoch_end=20,
                 n_patches=256,
                 iterations=90,
                 test_frequency=5,
                 validation_frequency=10):
        self.gen = gen
        self.F = F
        self.save_dir = save_dir
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.scheduler = scheduler
        self.epoch_start = epoch_start
        self.epoch_end = epoch_end
        self.n_patches = n_patches
        self.iterations = iterations
        self.test_frequency = test_frequency
        self.validation_frequency = validation_frequency
        
        # Create directories if they don't exist
        self.image_dir = os.path.join(save_dir, 'images')
        os.makedirs(self.image_dir, exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'checkpoints'), exist_ok=True)

        # Initialize optimizers
        self._initialize_optimizers()
        
    def _initialize_optimizers(self):
        """Initialize all optimizers with proper parameters"""
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
            lr=1e-4, betas=(0.5, 0.999)
        )

        self.optimizer_Dp_1 = torch.optim.Adam(
            self.gen.loss_a.discriminator.parameters(),
            lr=1e-4, betas=(0.5, 0.999)
        )
        self.optimizer_Dp_2 = torch.optim.Adam(
            self.gen.loss_b.discriminator.parameters(),
            lr=1e-4, betas=(0.5, 0.999)
        )
        self.optimizer_Du = torch.optim.Adam(
            self.gen.loss_c.discriminator.parameters(),
            lr=1e-4, betas=(0.5, 0.999)
        )
        try:
            self.optimizer_F = torch.optim.Adam(
                self.F.parameters(),
                lr=1e-4, betas=(0.5, 0.999)
            )
        except:
            self.optimizer_F = None

    def train_discriminators(self, x_p, y_p, x, y):
        """Train all discriminator networks"""
        self.gen.train()
        
        # Train Discriminator Dp1
        self.optimizer_Dp_1.zero_grad()
        s_xp = self.gen.encode_style(x_p, label=1)
        fake_xp, _, _ = self.gen(y_p, label=0, cross=True, s_given=s_xp)
        rec_xp, qloss_xp, _ = self.gen(x_p, label=1, cross=False)
        
        y2x_loss, _ = self.gen.loss_a(_, x_p, fake_xp, cond=x_p, optimizer_idx=1, 
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
        
        x2y_loss, _ = self.gen.loss_b(_, y_p, fake_yp_mixed, cond=y_p, 
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
        
        x2r_loss, _ = self.gen.loss_c(_, y, fake_y, optimizer_idx=1, 
                                     last_layer=None, split="train")
        y_rec_d_loss, _ = self.gen.loss_c(_, y, rec_y, optimizer_idx=1, 
                                         last_layer=None, split="train")
        
        disc_r_loss = 0.8*x2r_loss + 0.2*y_rec_d_loss
        disc_r_loss.backward()
        self.optimizer_Du.step()

        return fake_xp, fake_yp, fake_yp_mixed, fake_y, rec_xp, rec_yp, rec_y, \
               qloss_xp, qloss_yp, qloss_y, s_xp, s_yp, s_yr, s_r
    
    def generate_validation_images(self, x_p, y_p, x, y):
        """Generate images for validation visualization"""
        self.gen.eval()
        with torch.no_grad():
            # Generate supervised translations
            s_xp = self.gen.encode_style(x_p, label=1)
            s_yp = self.gen.encode_style(y_p, label=0)
            s_r = self.gen.encode_style(y, label=-1)
            s_yr = self.gen.style_mix(s_yp, s_r)
            
            fake_xp, _, _ = self.gen(y_p, label=0, cross=True, s_given=s_xp)
            fake_yp, _, _ = self.gen(x_p, label=1, cross=True, s_given=s_yp)
            fake_yp_mixed, _, _ = self.gen(x_p, label=1, cross=True, s_given=s_yr)
            
            # Generate unsupervised translations
            fake_y, _, _ = self.gen(x, label=1, cross=True, s_given=s_r)
            
            # Generate reconstructions
            rec_xp, _, _ = self.gen(x_p, label=1, cross=False)
            rec_yp, _, _ = self.gen(y_p, label=0, cross=False)
            rec_y, _, _ = self.gen(y, label=-1, cross=False)

        return {
            'supervised': {
                'x_p': x_p, 'y_p': y_p,
                'fake_xp': fake_xp, 'fake_yp': fake_yp,
                'fake_yp_mixed': fake_yp_mixed,
                'rec_xp': rec_xp, 'rec_yp': rec_yp
            },
            'unsupervised': {
                'x': x, 'y': y,
                'fake_y': fake_y,
                'rec_y': rec_y
            }
        }

    def save_validation_images(self, images, epoch, iteration):
        """Save validation image grids"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'epoch_{epoch}_iter_{iteration}_{timestamp}'
        
        # Create supervised grid
        supervised_grid = torch.cat([
            images['supervised']['x_p'],
            images['supervised']['y_p'],
            images['supervised']['fake_xp'],
            images['supervised']['fake_yp'],
            images['supervised']['fake_yp_mixed'],
            images['supervised']['rec_xp'],
            images['supervised']['rec_yp']
        ], dim=0)
        
        # Create unsupervised grid
        unsupervised_grid = torch.cat([
            images['unsupervised']['x'],
            images['unsupervised']['y'],
            images['unsupervised']['fake_y'],
            images['unsupervised']['rec_y']
        ], dim=0)
        
        # Save grids
        save_image(
            supervised_grid,
            os.path.join(self.image_dir, f'{filename}_supervised.png'),
            nrow=images['supervised']['x_p'].size(0),
            normalize=True
        )
        
        save_image(
            unsupervised_grid,
            os.path.join(self.image_dir, f'{filename}_unsupervised.png'),
            nrow=images['unsupervised']['x'].size(0),
            normalize=True
        )

    def validate(self, epoch, iteration):
        """Run validation step if validation loader is available"""
        if self.val_loader is None:
            return None, None, None
            
        try:
            # Get validation batch
            val_batch = next(iter(self.val_loader))
            x_p, y_p, x, y = val_batch.values()
            x_p, y_p = x_p.to(self.device), y_p.to(self.device)
            x, y = x.to(self.device), y.to(self.device)

            # Generate validation images and calculate losses
            val_images = self.generate_validation_images(x_p, y_p, x, y)
            if iteration % 10 == 0:  # Save images less frequently
                self.save_validation_images(val_images, epoch, iteration)

            # Calculate validation losses
            with torch.no_grad():
                lambda_sup = torch.cos(torch.tensor((torch.pi*(epoch - 1)/(self.epoch_end*2)))).to(self.device)
                
                l_supervised = calculate_supervised_loss(
                    self.gen, self.F, x_p, y_p, 
                    val_images['supervised']['fake_xp'],
                    val_images['supervised']['fake_yp'],
                    val_images['supervised']['fake_yp_mixed'],
                    val_images['supervised']['rec_xp'],
                    val_images['supervised']['rec_yp'],
                    None, None, None, None, None, None
                )

                l_unsupervised = calculate_unsupervised_loss(
                    self.gen, self.F, x, y,
                    val_images['unsupervised']['fake_y'],
                    val_images['unsupervised']['rec_y'],
                    None, self.n_patches, epoch
                )

                total_loss = calculate_total_loss(l_supervised, l_unsupervised, lambda_sup)

            return l_supervised.item(), l_unsupervised.item(), total_loss.item()
            
        except Exception as e:
            print(f"Validation error: {str(e)}")
            return None, None, None

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
        """Main training loop with optional validation and testing"""
        for epoch in range(self.epoch_start, self.epoch_end):
            lambda_sup = torch.cos(torch.tensor((torch.pi*(epoch - 1)/(self.epoch_end*2)))).to(self.device)
            
            for i in range(self.iterations):
                try:
                    # Get training data
                    batch = next(iter(self.train_loader))
                    x_p, y_p, x, y = batch.values()
                    x_p, y_p = x_p.to(self.device), y_p.to(self.device)
                    x, y = x.to(self.device), y.to(self.device)
                    
                    # Train discriminators and get generated images
                    outputs = self.train_discriminators(x_p, y_p, x, y)
                    fake_xp, fake_yp, fake_yp_mixed, fake_y, rec_xp, rec_yp, rec_y, \
                    qloss_xp, qloss_yp, qloss_y, s_xp, s_yp, s_yr, s_r = outputs
        
                    # Train Autoencoder
                    self.optimizer_AE.zero_grad()
        
                    # Calculate losses
                    l_supervised = calculate_supervised_loss(
                        self.gen, self.F, x_p, y_p, fake_xp, fake_yp, fake_yp_mixed,
                        rec_xp, rec_yp, s_xp, s_yp, s_yr, s_r, qloss_xp, qloss_yp
                    )
        
                    l_unsupervised = calculate_unsupervised_loss(
                        self.gen, self.F, x, y, fake_y, rec_y, qloss_y,
                        self.n_patches, epoch
                    )
        
                    loss = calculate_total_loss(l_supervised, l_unsupervised, lambda_sup)
        
                    loss.backward()
                    self.optimizer_AE.step()
                    
                    if self.optimizer_F is not None:
                        self.optimizer_F.step()
                    
                    # Run validation if available
                    if i % self.validation_frequency == 0:
                        val_results = self.validate(epoch, i)
                        if all(v is not None for v in val_results):
                            val_supervised, val_unsupervised, val_total = val_results
                            print(f"\nValidation Losses - Supervised: {val_supervised:.4f}, "
                                  f"Unsupervised: {val_unsupervised:.4f}, "
                                  f"Total: {val_total:.4f}")
                    
                    # Print training progress
                    if i % 1 == 0:
                        print(f"Epoch [{epoch}/{self.epoch_end}], "
                              f"Iteration [{i}/{self.iterations}], "
                              f"Loss: {loss.item():.4f}")
                        print(f"Loss supervised: {l_supervised.item():.4f}, "
                              f"Loss unsupervised: {l_unsupervised.item():.4f}")
                    
                    torch.cuda.empty_cache()
                    
                except Exception as e:
                    print(f"Error during training iteration: {str(e)}")
                    continue
            
            # Save checkpoint
            if (epoch + 1) % self.test_frequency == 0:
                self.save_checkpoint(epoch, i)
            
            if self.scheduler is not None:
                self.scheduler.step()