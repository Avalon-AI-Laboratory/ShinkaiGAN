import os
import torch
import torch.nn as nn
import functools
from torchvision import transforms
import argparse
from torch.utils.data import DataLoader

from modules.modular.trainer import Trainer
from models.CUT.resnetG import ResnetGenerator
from models.vqi2i.modules.mlp_sampler import PatchSampleF
from models.vqi2i.vqgan_model.vqi2i_adain import VQI2ICrossGAN_AdaIN
from data_modules.datasetClass import myDataset

def parse_args():
    parser = argparse.ArgumentParser(description='Train Style Transfer Model')
    
    # Data paths
    parser.add_argument('--train_dir', type=str, required=True,
                        help='Path to training data directory')
    parser.add_argument('--val_dir', type=str, required=True,
                        help='Path to validation data directory')
    parser.add_argument('--test_dir', type=str, required=True,
                        help='Path to test data directory')
    
    # Training settings
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs to train')
    parser.add_argument('--iterations', type=int, default=60000,
                        help='Number of iterations per epoch')
    parser.add_argument('--n_patches', type=int, default=256,
                        help='Number of patches for training')
    parser.add_argument('--test_frequency', type=int, default=10,
                        help='How often to run test evaluation (epochs)')
    
    # Model settings - Updated to match ResnetGenerator parameters
    parser.add_argument('--image_size', type=int, default=256,
                        help='Size of input images')
    parser.add_argument('--input_nc', type=int, default=3,
                        help='Number of input image channels')
    parser.add_argument('--output_nc', type=int, default=3,
                        help='Number of output image channels')
    parser.add_argument('--ngf', type=int, default=64,
                        help='Number of generator filters')
    parser.add_argument('--n_blocks', type=int, default=9,
                        help='Number of ResNet blocks')
    parser.add_argument('--norm_layer', type=str, default='instance',
                        help='Type of normalization layer: batch | instance')
    parser.add_argument('--use_dropout', action='store_true',
                        help='Use dropout in ResNet generator')
    parser.add_argument('--padding_type', type=str, default='reflect',
                        help='Padding type for ResNet: reflect | replicate | zero')
    parser.add_argument('--no_antialias', action='store_true',
                        help='Disable antialiasing in generator')
    parser.add_argument('--no_antialias_up', action='store_true',
                        help='Disable antialiasing in generator upsampling')
    
    # PatchSampleF settings - Updated to match provided implementation
    parser.add_argument('--use_mlp', action='store_true',
                        help='Use MLP in PatchSampleF')
    parser.add_argument('--nc_mlp', type=int, default=256,
                        help='Size of MLP hidden layers')
    parser.add_argument('--mlp_init_type', type=str, default='normal',
                        help='MLP initialization type')
    parser.add_argument('--mlp_init_gain', type=float, default=0.02,
                        help='MLP initialization gain')
    
    # Optimization settings
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--beta1', type=float, default=0.5,
                        help='Beta1 for Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.999,
                        help='Beta2 for Adam optimizer')
    
    # System settings
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='GPU ID to use')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers for data loading')
    parser.add_argument('--save_dir', type=str, default='./results',
                        help='Directory to save results')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    
    return parser.parse_args()

def setup_dataloaders(args):
    """Setup data transformations and create dataloaders"""
    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], 
                           std=[0.5, 0.5, 0.5])
    ])
    
    train_dataset = myDataset(
        root_dir=args.train_dir,
        transform=transform,
        is_train=True
    )
    
    val_dataset = myDataset(
        root_dir=args.val_dir,
        transform=transform,
        is_train=False
    )
    
    test_dataset = myDataset(
        root_dir=args.test_dir,
        transform=transform,
        is_train=False
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader

def setup_models(args, device):
    """Initialize models with correct parameters"""
    # Get normalization layer
    if args.norm_layer == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    else:  # instance normalization
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    
    # Initialize ResnetGenerator with the correct parameters
    generator = ResnetGenerator(
        input_nc=args.input_nc,
        output_nc=args.output_nc,
        ngf=args.ngf,
        norm_layer=norm_layer,
        use_dropout=args.use_dropout,
        n_blocks=args.n_blocks,
        padding_type=args.padding_type,
        no_antialias=args.no_antialias,
        no_antialias_up=args.no_antialias_up,
        # opt=args.opt
    ).to(device)
    
    # Initialize PatchSampleF with the correct parameters
    feature_extractor = PatchSampleF(
        use_mlp=args.use_mlp,
        init_type=args.mlp_init_type,
        init_gain=args.mlp_init_gain,
        nc=args.nc_mlp,
        gpu_ids=[args.gpu_id]
    ).to(device)
    
    # Initialize VQI2ICrossGAN_AdaIN
    vqi2i_model = VQI2ICrossGAN_AdaIN(
        style_dim=args.style_dim,
        n_patches=args.n_patches,
        device=device
    ).to(device)
    
    return generator, feature_extractor, vqi2i_model

def main():
    args = parse_args()
    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    print("Setting up data loaders...")
    train_loader, val_loader, test_loader = setup_dataloaders(args)
    
    print("Initializing models...")
    generator, feature_extractor, vqi2i_model = setup_models(args, device)
    
    print("Initializing trainer...")
    trainer = Trainer(
        gen=generator,
        F=feature_extractor,
        vqi2i=vqi2i_model,
        save_dir=args.save_dir,
        device=device,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        scheduler=None,  # Will be setup in Trainer
        epoch_start=0,
        epoch_end=args.epochs,
        n_patches=args.n_patches,
        iterations=args.iterations,
        test_frequency=args.test_frequency
    )
    
    if args.resume is not None:
        print(f"Resuming from checkpoint: {args.resume}")
        start_epoch, start_iteration = trainer.load_checkpoint(args.resume)
        trainer.epoch_start = start_epoch
    
    print("Starting training...")
    trainer.train()
    
    print("Running final evaluation...")
    final_metrics = trainer.evaluate()
    print("\nFinal Test Metrics:")
    for domain, metrics in final_metrics.items():
        print(f"\n{domain}:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")

if __name__ == '__main__':
    main()