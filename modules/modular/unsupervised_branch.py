import torch
from models.vqi2i.modules.losses.SRC import calculate_R_loss
from models.vqi2i.modules.losses.patchHDCE import calculate_HDCE_loss

class UnsupervisedTrainingBranch:
    def __init__(self, gen, device, F, n_patches):
        self.gen = gen
        self.device = device
        self.F = F
        self.n_patches = n_patches

    def forward(self, x, fake_y):
        c_y_fake, c_y_fake_q = self.gen.encode_content(fake_y)
        c_x, c_x_quantized = self.gen.encode_content(x)
        L_cP = torch.mean(torch.abs(c_y_fake.detach() - c_x)).to(self.device)
        L_cP_q = torch.mean(torch.abs(c_y_fake_q.detach() - c_x_quantized)).to(self.device)

        L_sP = self.calculate_style_loss(x, fake_y)
        l_src, weight = calculate_R_loss(x, fake_y, [0, 4, 9, 12], self.gen, self.F, self.n_patches)
        l_hdce = calculate_HDCE_loss(x, fake_y, weight, [0, 4, 9, 12], self.gen, self.F)

        return L_cP + L_cP_q + L_sP + 0.5 * (l_src + l_hdce)

    def calculate_style_loss(self, x, fake_y):
        s_x_fake = self.gen.encode_style(fake_y, label=1)
        s_x = self.gen.encode_style(x, label=1)
        L_sP = torch.mean(torch.abs(s_x_fake.detach() - s_x)).to(self.device)
        return L_sP
