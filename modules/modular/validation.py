import os
import torch
import torchvision
from torchvision.utils import save_image
import matplotlib.pyplot as plt
from datetime import datetime

class Validator:
    def __init__(self, gen, F, device, val_loader, save_dir):
        """
        Initialize the validator
        
        Args:
            gen: Generator model
            F: Feature network
            device: Computing device (CPU/GPU)
            val_loader: Validation data loader
            save_dir: Directory to save outputs
        """
        self.gen = gen
        self.F = F
        self.device = device
        self.val_loader = val_loader
        self.save_dir = save_dir
        
        # Create directories for saving results
        self.checkpoint_dir = os.path.join(save_dir, 'checkpoints')
        self.image_dir = os.path.join(save_dir, 'validation_images')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.image_dir, exist_ok=True)

    def save_model(self, epoch, iteration, optimizer_AE, optimizer_F, 
                  optimizer_Dp_1, optimizer_Dp_2, optimizer_Du):
        """
        Save model checkpoints
        """
        checkpoint = {
            'epoch': epoch,
            'iteration': iteration,
            'gen_state_dict': self.gen.state_dict(),
            'F_state_dict': self.F.state_dict(),
            'optimizer_AE_state_dict': optimizer_AE.state_dict(),
            'optimizer_F_state_dict': optimizer_F.state_dict() if optimizer_F else None,
            'optimizer_Dp_1_state_dict': optimizer_Dp_1.state_dict(),
            'optimizer_Dp_2_state_dict': optimizer_Dp_2.state_dict(),
            'optimizer_Du_state_dict': optimizer_Du.state_dict()
        }
        
        # Save latest checkpoint
        torch.save(checkpoint, 
                  os.path.join(self.checkpoint_dir, 'latest_checkpoint.pth'))
        
        # Save periodic checkpoint
        checkpoint_path = os.path.join(
            self.checkpoint_dir, 
            f'checkpoint_epoch_{epoch}.pth'
        )
        torch.save(checkpoint, checkpoint_path)
        print(f"Model checkpoint saved at {checkpoint_path}")

    def generate_validation_images(self, x_p, y_p, x, y):
        """
        Generate images for validation visualization
        """
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

    def create_image_grid(self, images, filename):
        """
        Create and save a grid of images
        """
        # Combine images into grids
        supervised_grid = torch.cat([
            images['supervised']['x_p'],
            images['supervised']['y_p'],
            images['supervised']['fake_xp'],
            images['supervised']['fake_yp'],
            images['supervised']['fake_yp_mixed'],
            images['supervised']['rec_xp'],
            images['supervised']['rec_yp']
        ], dim=0)
        
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

    def validate(self, epoch, iteration, save_model=False, **optimizers):
        """
        Run validation and optionally save model checkpoint
        
        Args:
            epoch: Current epoch number
            iteration: Current iteration number
            save_model: Whether to save model checkpoint
            **optimizers: Dictionary containing optimizer objects
        """
        # Get validation batch
        val_batch = next(iter(self.val_loader))
        x_p, y_p, x, y = val_batch.values()
        x_p, y_p = x_p.to(self.device), y_p.to(self.device)
        x, y = x.to(self.device), y.to(self.device)

        # Generate validation images
        val_images = self.generate_validation_images(x_p, y_p, x, y)
        
        # Create filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'epoch_{epoch}_iter_{iteration}_{timestamp}'
        
        # Save images
        self.create_image_grid(val_images, filename)
        
        # Save model checkpoint if requested
        if save_model:
            self.save_model(
                epoch, iteration,
                optimizers['optimizer_AE'],
                optimizers['optimizer_F'],
                optimizers['optimizer_Dp_1'],
                optimizers['optimizer_Dp_2'],
                optimizers['optimizer_Du']
            )

        self.gen.train()  # Switch back to training mode