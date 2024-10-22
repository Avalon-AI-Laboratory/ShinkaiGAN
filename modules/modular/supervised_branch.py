import torch
from autoencoder import AutoEncoder

class SupervisedTrainingBranch:
    def __init__(self, gen, device):
        self.gen = gen
        self.device = device
        self.autoencoder = AutoEncoder(gen, device)

    def forward(self, x_p, y_p):
        s_xp = self.gen.encode_style(x_p, label=1)
        fake_xp, _, _ = self.gen(y_p, label=0, cross=True, s_given=s_xp)

        (rec_xp, qloss_xp), (rec_yp, qloss_yp), _, s_xp, s_yp, _ = self.autoencoder.forward(x_p, y_p, None)
        ae_lossA, ae_lossB, _ = self.autoencoder.calculate_losses(rec_xp, rec_yp, None, qloss_xp, qloss_yp, None)

        content_loss, style_loss = self.calculate_content_style_loss(x_p, y_p, rec_xp, fake_xp, s_xp, s_yp)

        return ae_lossA + ae_lossB + 0.5 * (content_loss + style_loss)

    def calculate_content_style_loss(self, x_p, y_p, rec_xp, fake_xp, s_xp, s_yp):
        c_xp, c_xp_quantized = self.gen.encode_content(x_p)
        c_yp, c_yp_quantized = self.gen.encode_content(fake_xp)
        content_loss = torch.mean(torch.abs(c_xp.detach() - c_yp)).to(self.device)
        content_quantized_loss = torch.mean(torch.abs(c_xp_quantized.detach() - c_yp_quantized)).to(self.device)

        style_loss_xp = torch.mean(torch.abs(s_xp.detach() - s_xp)).to(self.device)
        style_loss_yp = torch.mean(torch.abs(s_yp.detach() - s_yp)).to(self.device)

        return 0.5 * (content_loss + content_quantized_loss), 0.3 * (style_loss_xp + style_loss_yp)
