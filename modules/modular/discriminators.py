class Discriminators:
    def __init__(self, gen):
        self.gen = gen

    def forward_Dp1(self, x_p, fake_xp):
        y2x_loss, log = self.gen.loss_a(None, x_p, fake_xp, cond=x_p, optimizer_idx=1)
        return y2x_loss

    def forward_Dp2(self, y_p, fake_yp):
        x2y_loss, log = self.gen.loss_b(None, y_p, fake_yp, cond=y_p, optimizer_idx=1)
        return x2y_loss

    def forward_Du(self, y, fake_y):
        x2r_loss, log = self.gen.loss_c(None, y, fake_y, optimizer_idx=1)
        return x2r_loss
