import torch
from modular.trainer import Trainer
from models.vqi2i.modules import YourGenClass

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gen = YourGenClass().to(device)

    # Load data, initialize other components like F, train_loader, val_loader, etc.
    trainer = Trainer(gen, F, device, train_loader, val_loader, scheduler, epoch_start=1, epoch_end=100)
    trainer.train()

if __name__ == "__main__":
    main()
