import torch
import os
from tqdm import tqdm
import numpy as np
from torchvision.utils import save_image

class Tester:
    def __init__(self,
                 gen,
                 F,
                 device,
                 test_loader,
                 save_dir,
                 n_samples=5):
        """
        Initialize the Tester class.
        
        Args:
            gen: Generator model
            F: Feature extractor model
            device: Computing device (CPU/GPU)
            test_loader: DataLoader for test data
            save_dir: Directory to save test results
            n_samples: Number of sample images to save for visual inspection
        """
        self.gen = gen
        self.F = F
        self.device = device
        self.test_loader = test_loader
        self.save_dir = save_dir
        self.n_samples = n_samples
        
        # Create directories for saving results
        self.test_dir = os.path.join(save_dir, 'test_results')
        self.sample_dir = os.path.join(self.test_dir, 'samples')
        os.makedirs(self.test_dir, exist_ok=True)
        os.makedirs(self.sample_dir, exist_ok=True)

    @torch.no_grad()
    def compute_metrics(self, real_img, generated_img):
        """
        Compute evaluation metrics between real and generated images.
        
        Args:
            real_img: Ground truth images
            generated_img: Generated images
            
        Returns:
            Dictionary containing computed metrics
        """
        # Extract features using F
        real_features = self.F(real_img)
        gen_features = self.F(generated_img)
        
        # Compute L1 distance between features
        feature_distance = torch.mean(torch.abs(real_features - gen_features))
        
        # Compute PSNR
        mse = torch.mean((real_img - generated_img) ** 2)
        psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
        
        return {
            'feature_distance': feature_distance.item(),
            'psnr': psnr.item(),
            'mse': mse.item()
        }

    def save_sample_images(self, x_p, y_p, x, y, fake_xp, fake_yp, fake_yp_mixed, 
                          fake_y, rec_xp, rec_yp, rec_y, batch_idx):
        """
        Save sample images for visual inspection.
        
        Args:
            Various input and generated images
            batch_idx: Current batch index
        """
        if batch_idx >= self.n_samples:
            return
            
        # Create grid of images
        imgs = torch.stack([
            x_p[0], y_p[0],  # Original paired images
            fake_xp[0], fake_yp[0],  # Cross-domain translations
            fake_yp_mixed[0],  # Mixed style translation
            rec_xp[0], rec_yp[0],  # Reconstructions of paired images
            x[0], y[0],  # Original unpaired images
            fake_y[0], rec_y[0]  # Unpaired translations and reconstructions
        ])
        
        save_image(imgs, 
                  os.path.join(self.sample_dir, f'sample_{batch_idx}.png'),
                  nrow=4,
                  normalize=True,
                  range=(-1, 1))

    @torch.no_grad()
    def test(self, checkpoint_path=None):
        """
        Run full testing procedure.
        
        Args:
            checkpoint_path: Optional path to model checkpoint to load
            
        Returns:
            Dictionary containing average metrics across test set
        """
        # Load checkpoint if provided
        if checkpoint_path is not None:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.gen.load_state_dict(checkpoint['generator_state_dict'])
            self.F.load_state_dict(checkpoint['feature_extractor_state_dict'])
        
        self.gen.eval()
        self.F.eval()
        
        metrics_list = []
        print("Starting test evaluation...")
        
        for batch_idx, batch in enumerate(tqdm(self.test_loader)):
            x_p, y_p, x, y = batch.values()
            x_p, y_p = x_p.to(self.device), y_p.to(self.device)
            x, y = x.to(self.device), y.to(self.device)
            
            # Generate all variations as in training
            s_xp = self.gen.encode_style(x_p, label=1)
            s_yp = self.gen.encode_style(y_p, label=0)
            s_r = self.gen.encode_style(y, label=-1)
            s_yr = self.gen.style_mix(s_yp, s_r)
            
            # Generate images
            fake_xp, _, _ = self.gen(y_p, label=0, cross=True, s_given=s_xp)
            fake_yp, _, _ = self.gen(x_p, label=1, cross=True, s_given=s_yp)
            fake_yp_mixed, _, _ = self.gen(x_p, label=1, cross=True, s_given=s_yr)
            fake_y, _, _ = self.gen(x, label=1, cross=True, s_given=s_r)
            
            rec_xp, _, _ = self.gen(x_p, label=1, cross=False)
            rec_yp, _, _ = self.gen(y_p, label=0, cross=False)
            rec_y, _, _ = self.gen(y, label=-1, cross=False)
            
            # Compute metrics for each type of generation
            metrics = {
                'paired_x': self.compute_metrics(x_p, rec_xp),
                'paired_y': self.compute_metrics(y_p, rec_yp),
                'unpaired': self.compute_metrics(y, rec_y),
                'cross_domain_x': self.compute_metrics(x_p, fake_xp),
                'cross_domain_y': self.compute_metrics(y_p, fake_yp),
                'mixed_style': self.compute_metrics(y_p, fake_yp_mixed)
            }
            
            metrics_list.append(metrics)
            
            # Save sample images
            self.save_sample_images(x_p, y_p, x, y, fake_xp, fake_yp, fake_yp_mixed,
                                 fake_y, rec_xp, rec_yp, rec_y, batch_idx)
        
        # Compute average metrics
        avg_metrics = {}
        for domain in metrics_list[0].keys():
            avg_metrics[domain] = {}
            for metric in metrics_list[0][domain].keys():
                values = [m[domain][metric] for m in metrics_list]
                avg_metrics[domain][metric] = np.mean(values)
        
        # Save metrics to file
        with open(os.path.join(self.test_dir, 'test_metrics.txt'), 'w') as f:
            for domain, metrics in avg_metrics.items():
                f.write(f"\n{domain}:\n")
                for metric, value in metrics.items():
                    f.write(f"{metric}: {value:.4f}\n")
        
        return avg_metrics

    def test_single_image(self, image_path, save_path=None):
        """
        Test the model on a single image.
        
        Args:
            image_path: Path to input image
            save_path: Optional path to save results
            
        Returns:
            Dictionary containing generated images
        """
        self.gen.eval()
        self.F.eval()
        
        # Load and preprocess image
        from PIL import Image
        from torchvision import transforms
        
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        image = Image.open(image_path).convert('RGB')
        image = transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # Generate different style variations
            s_r = self.gen.encode_style(image, label=-1)
            fake_y, _, _ = self.gen(image, label=1, cross=True, s_given=s_r)
            rec_image, _, _ = self.gen(image, label=-1, cross=False)
            
            results = {
                'original': image,
                'reconstruction': rec_image,
                'style_transfer': fake_y
            }
            
            if save_path is not None:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                save_image(torch.cat(list(results.values())), 
                          save_path,
                          nrow=len(results),
                          normalize=True,
                          range=(-1, 1))
        
        return results