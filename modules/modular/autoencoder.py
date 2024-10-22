import torch

class AutoEncoder:
    def __init__(self, gen, device):
        self.gen = gen
        self.device = device

    def forward(self, x_p, y_p, y):
        s_xp = self.gen.encode_style(x_p, label=1)
        rec_xp, qloss_xp, _ = self.gen(x_p, label=1, cross=False)
        
        s_yp = self.gen.encode_style(y_p, label=0)
        rec_yp, qloss_yp, _ = self.gen(y_p, label=0, cross=False)

        s_r = self.gen.encode_style(y, label=-1)
        rec_y, qloss_y, _ = self.gen(y, label=1, cross=True, s_given=s_r)

        return (rec_xp, qloss_xp), (rec_yp, qloss_yp), (rec_y, qloss_y), s_xp, s_yp, s_r

    def calculate_losses(self, rec_xp, rec_yp, rec_y, qloss_xp, qloss_yp, qloss_y):
        ae_lossA, _ = self.gen.loss_a(qloss_xp, rec_xp, switch_weight=0.1)
        ae_lossB, _ = self.gen.loss_b(qloss_yp, rec_yp, switch_weight=0.1)
        ae_lossC, _ = self.gen.loss_c(qloss_y, rec_y, switch_weight=0.1)

        return ae_lossA, ae_lossB, ae_lossC
