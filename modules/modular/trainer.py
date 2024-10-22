import torch
from supervised_branch import SupervisedTrainingBranch
from unsupervised_branch import UnsupervisedTrainingBranch
from total_loss import TotalLoss

class Trainer:
    def __init__(self, gen, F, device, train_loader, val_loader, scheduler, epoch_start, epoch_end, n_patches=256, iterations=60000):
        self.gen = gen
        self.F = F
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.scheduler = scheduler
        self.epoch_start = epoch_start
        self.epoch_end = epoch_end
        self.n_patches = n_patches
        self.iterations = iterations

        self.supervised_branch = SupervisedTrainingBranch(gen, device)
        self.unsupervised_branch = UnsupervisedTrainingBranch(gen, device, F, n_patches)
        self.total_loss_calculator = TotalLoss(self.supervised_branch, self.unsupervised_branch)

    def train(self):
        for epoch in range(self.epoch_start, self.epoch_end):
            lambda_sup = torch.cos(torch.tensor((torch.pi*(epoch - 1)/(self.epoch_end*2)))).to(self.device)
            for i in range(self.iterations):
                x_p, y_p, x, y = next(iter(self.train_loader)).values()
                x_p, y_p, x, y = x_p.to(self.device), y_p.to(self.device), x.to(self.device), y.to(self.device)

                s_r = self.gen.encode_style(y, label=-1)
                fake_y, _, _ = self.gen(x, label=1, cross=True, s_given=s_r)

                total_loss = self.total_loss_calculator.calculate_total_loss(x_p, y_p, x, y, fake_y, lambda_sup)
                total_loss.backward()
                self.scheduler.step()

    def validate(self):
        # Implement validation logic
        pass
