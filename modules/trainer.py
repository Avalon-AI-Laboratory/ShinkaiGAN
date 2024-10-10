#### WARNING: THIS CODE STILL IN PROGRESS! ####
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import random_split, DataLoader, Dataset
import torch.nn.functional as F
import torchvision.transforms as T
import os
import random
import numpy as np

from models.vqi2i.vqgan_model.vqi2i_adain import VQI2ICrossGAN_AdaIN
from models.vqi2i.modules.discriminators.nlayers_disc import NLayerDiscriminator

class PairDataset(Dataset):
    def __init__(self, root, mode, resize=256, cropsize=256, hflip=0.0):
        self.root = root
        self.mode = mode
        src_imgs = os.listdir(os.path.join(self.root, "src"))
        style_ref_imgs = os.listdir(os.path.join(self.root, "style_ref"))
        self.src = [x for x in src_imgs]
        self.src_size = len(self.src)

        self.style_ref = [x for x in style_ref_imgs]
        self.style_ref_size = len(self.style_ref)

        self.input_dim_src, self.input_dim_tgt = 3, 3

        transforms = [T.Resize((resize, resize), Image.BICUBIC)]

        if mode == "train":
            transforms.append(T.RandomCrop(cropsize))
        else:
            transforms.append(T.CenterCrop(cropsize))
        
        # Flip
        transforms.append(T.RandomHorizontalFlip(p=1.0))

        transforms.append(T.ToTensor())
        transforms.append(T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))
        self.transforms = T.Compose(transforms)

        transforms_no_flip = [T.Resize((resize, resize), Image.BICUBIC)]

        if mode == "train":
            transforms_no_flip.append(T.RandomCrop(cropsize))
        else:
            transforms_no_flip.append(T.CenterCrop(cropsize))
        
        transforms_no_flip.append(T.ToTensor())
        transforms_no_flip.append(T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))
        self.transforms_no_flip = T.Compose(transforms_no_flip)
    
    def __getitem__(self, index):
        flip_or_not = random.random()
        random_style_index = np.random.randint(0, self.style_ref_size)

        A = os.path.join(self.root, self.mode + "_src", self.src[index])
        data_A = self.load_img(A, self.input_dim_src, flip_or_not)

        B = os.path.join(self.root, self.mode + "_tgt", self.src[index])
        data_B = self.load_img(B, self.input_dim_tgt, flip_or_not)

        style_ref = os.path.join(self.root, "style_ref", self.style_ref[random_style_index])
        data_style_ref = self.load_img(style_ref, self.input_dim_tgt, 0.0)

        return data_A, data_B, data_style_ref

    def __len__(self):
        return self.src_size

    def load_img(self, img_name, input_dim, flip_or_not):
        img = Image.open(img_name).convert("RGB")
        img = self.transforms(img) if flip_or_not > 0.5 else self.transforms_no_flip(img)

        if(input_dim == 1):
            img = img[0, ...] * 0.299 + img[1, ...] * 0.587 + img[2, ...] * 0.114
            img = img.unsqueeze(0)
        return img

class Trainer:
    def __init__(self, 
                 root_dir, 
                 save_path,
                 n_e=128, 
                 embed_dim=128,
                 z_channels=128, 
                 epoch_start=1, 
                 epoch_end=20,
                 batch_size=128,
                 learning_rate=1e-5,
                 img_size=256,
                 switch_weight=0.1,
                 device="cuda"
                 ):
        self.root_dir = root_dir
        self.save_path = save_path
        train_data = PairDataset(root_dir, "train", img_size, img_size)
        self.train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=8)
        self.model = VQI2ICrossGAN_AdaIN(
            n_embed = n_e,
            embed_dim=embed_dim,
            z_channels=z_channels,
            resolution=img_size
        ).to(device)

        self.device = device
        self.disc_a = NLayerDiscriminator(input_nc=3, ndf=64)
        self.disc_b = NLayerDiscriminator(input_nc=3, ndf=64)

        self.epoch_start = epoch_start
        self.epoch_end = epoch_end
        
        self.optimizer_ae = torch.optim.Adam(list(self.model.content_enc.parameters())+ \
                                             list(self.model.decoder_a.parameters())+ \
                                             list(self.model.decoder_b.parameters())+ \
                                             list(self.model.quantize.parameters())+ \
                                             list(self.model.quant_conv.parameters())+ \
                                             list(self.model.post_quant_conv.parameters())+ \
                                             list(self.model.style_enc_a.parameters())+ \
                                             list(self.model.style_enc_b.parameters()), 
                                             lr=learning_rate, betas=(0.5, 0.999))
        
        #### BAGIAN INI BELUM TERIMPLEMENTASI ############
        self.opt_disc_a = torch.optim.Adam(self.disc_a.parameters(), lr=learning_rate, betas=(0.5, 0.999))
        self.opt_disc_b = torch.optim.Adam(self.disc_b.parameters(), lr=learning_rate, betas=(0.5, 0.999))
        ##################################################

        self.train_ae_a_error = []
        self.train_ae_b_error = []
        self.train_disc_a_error = []
        self.train_disc_b_error = []
        self.train_disc_a2b_error = []
        self.train_disc_b2a_error = []
        self.train_res_rec_error = []

        self.train_style_a_error = []
        self.train_style_b_error = []
        self.train_content_loss = []
        self.train_cross_recon_loss = []

        self.iterations = len(train_data) // batch_size
        self.iterations = self.iterations + 1 if len(train_data) % batch_size != 0 else self.iterations

    
    def train(self):
        self.model.train()

        for epoch in range(self.epoch_start, self.epoch_end):
            for i in range(self.iterations):
                dataA, dataB, dataR = next(iter(self.train_loader))
                dataA, dataB, dataR = dataA.to(self.device), dataB.to(self.device), dataR.to(self.device)

                ########## SUPERVISED TRAINING BRANCH ##########

                # Latih diskriminator A
                self.opt_disc_a.zero_grad()

                # Note, label ini artinya apa? Sudah kejawab

                s_x = self.model.encode_style(dataA, label=1)
                fake_A, _, _ = self.model(dataB, label=0, cross=True, s_given=s_x)

                rec_A, qlossA, _ = self.model(dataA, label=1, cross=False) # Implementasi model untuk merekonstruksi data dirinya sendiri

                #### WARNING, BAGIAN INI AKAN PERLU DIPERBAIKI #####
                b2a_loss, log = self.model.loss_a(_, dataA, fake_A, optimizer_idx=1, last_layer=None, split="train")
                a_rec_d_loss, _ = self.model.loss_a(_, dataA, rec_A, optimizer_idx=1, last_layer=None, split="train")



                ####################################################

                disc_a_loss = 0.8*b2a_loss + 0.2*a_rec_d_loss
                disc_a_loss.backward()
                self.opt_disc_a.step()

                # Latih diskrminator B
                self.opt_disc_b.zero_grad()

                s_y = self.model.encode_style(dataB, label=0)
                s_r = self.model.encode_style(dataR, label=-1)
                s_y_r = self.model.style_mix(s_y, s_r)
                fake_B, _, s_y_r_sampled = self.model(dataA, label=1, cross=True, s_given=s_y_r)

                rec_B, qlossB, _ = self.model(dataB, label=0, cross=False)

                #### WARNING, BAGIAN INI AKAN PERLU DIPERBAIKI #####
                a2b_loss, log = self.model.loss_b(_, dataB, fake_B, optimizer_idx=1, last_layer=None, split="train")
                b_rec_d_loss, _ = self.model.loss_b(_, dataB, rec_B, optimizer_idx=1, last_layer=None, split="train")
                ####################################################

                disc_b_loss = 0.8*a2b_loss + 0.2*b_rec_d_loss
                disc_b_loss.backward()
                self.opt_disc_b.step()

                # Latih autoencoder
                self.optimizer_ae.zero_grad()

                ae_lossA, _ = self.model.loss_a(qlossA, dataA, rec_A, fake=fake_A, switch_weight=0.1, optimizer_idx=0, last_layer=self.model.get_last_layer(label=1), split="train")

                # Cross path with style A (s_x)
                AtoBtoA, _, s_a_from_cross = self.model(fake_A, label=1, cross=False)

                ae_lossB, _ = self.model.loss_b(qlossB, dataB, rec_B, fake=fake_B, switch_weight=0.1, optimizer_idx=0, last_layer=self.model.get_last_layer(label=0), split="train")
                BtoAtoB, _, s_b_from_cross = self.model(fake_B, label=0, cross=False)

                # Style loss
                style_loss_a = torch.mean(torch.abs(s_x.detach() - s_a_from_cross)).to(self.device)
                style_loss_b = torch.mean(torch.abs(s_y.detach() - s_b_from_cross)).to(self.device) # bagian ini perlu diperbaiki
                style_loss_abr = torch.mean(torch.abs(s_y_r.detach() - s_y_r_from_cross)).to(self.device) # bagian ini perlu juga diperbaiki
                style_loss  = 0.5 * (style_loss_a + style_loss_b + style_loss_abr)

                # Content loss
                c_a, c_a_quantized = self.model.encode_content(dataA)
                c_b, c_b_quantized = self.model.encode_content(dataB)
                content_loss = torch.mean(torch.abs(c_a.detach() - c_b)).to(self.device)
                content_quantized_loss = torch.mean(torch.abs(c_a_quantized.detach() - c_b_quantized)).to(self.device)  
                content_loss = 0.5 * (content_loss + content_quantized_loss)

                # Cross reconstruction loss
                cross_recons_loss_a = torch.mean(torch.abs(dataA.detach() - fake_A)).to(self.device)
                cross_recons_loss_b = torch.mean(torch.abs(dataB.detach() - fake_B)).to(self.device)
                cross_recons_loss = 0.5 * (cross_recons_loss_a + cross_recons_loss_b)

                gen_loss = ae_lossA + ae_lossB + 0.5 * (style_loss + content_loss) + 3.0*cross_recons_loss # Ini perlu dievaluasi, karena sistem kita tidak pixel-to-pixel
                gen_loss.backward()
                self.optimizer_ae.step()

                data = torch.cat((dataA, dataB), 0).to(self.device)
                rec = torch.cat((rec_A, rec_B), 0).to(self.device)
                recon_error = F.mse_loss(rec, data)

                self.train_res_rec_error.append(recon_error.item())
                self.train_ae_a_error.append(ae_lossA.item())
                self.train_ae_b_error.append(ae_lossB.item())

                self.train_disc_a_error.append(disc_a_loss.item())
                self.train_disc_b_error.append(disc_b_loss.item())
                self.train_disc_a2b_error.append(a2b_loss.item())
                self.train_disc_b2a_error.append(b2a_loss.item())

                self.train_style_a_error.append(style_loss_a.item())
                self.train_style_b_error.append(style_loss_b.item())

                self.train_content_loss.append(content_loss.item())
                self.train_cross_recon_loss.append(cross_recons_loss.item())

                if (i + 1) % 1000 == 0:
                    _rec  = 'epoch {}, {} iterations\n'.format(epoch, i+1)
                    _rec += '(A domain) ae_loss: {:8f}, disc_loss: {:8f}\n'.format(
                                np.mean(self.train_ae_a_error[-1000:]), np.mean(self.train_disc_a_error[-1000:]))
                    _rec += '(B domain) ae_loss: {:8f}, disc_loss: {:8f}\n'.format(
                                np.mean(self.train_ae_b_error[-1000:]), np.mean(self.train_disc_b_error[-1000:]))
                    _rec += 'A vs A2B loss: {:8f}, B vs B2A loss: {:8f}\n'.format(
                                np.mean(self.train_disc_a2b_error[-1000:]), np.mean(self.train_disc_b2a_error[-1000:]))
                    _rec += 'recon_error: {:8f}\n\n'.format(
                        np.mean(self.train_res_rec_error[-1000:]))
                    
                    _rec += 'style_a_loss: {:8f}\n\n'.format(
                        np.mean(self.train_style_a_loss[-1000:]))
                    _rec += 'style_b_loss: {:8f}\n\n'.format(
                        np.mean(self.train_style_b_loss[-1000:]))
                    
                    _rec += 'content_loss: {:8f}\n\n'.format(
                        np.mean(self.train_content_loss[-1000:]))

                    _rec += 'cross_recons_loss: {:8f}\n\n'.format(
                        np.mean(self.train_cross_recons_loss[-1000:]))
        
                    print(_rec)
                    with open(os.path.join(os.getcwd(), self.save_path, "loss.txt"), 'a') as f:
                        f.write(_rec)
                        f.close()
        torch.save(
            {
                'model_state_dict': self.model.state_dict(),
                'opt_ae_state_dict': self.optimizer_ae.state_dict(),
                'opt_disc_a_state_dict': self.opt_disc_a.state_dict(),
                'opt_disc_b_state_dict': self.opt_disc_b.state_dict()
            }, os.path.join(os.getcwd(), self.save_path, 'settingc_latest.pt'))
        
        ########## UNSUPERVISED TRAINING BRANCH ##########

        if(epoch % 20 == 0 and epoch >= 20):
            torch.save(
                {
                    'model_state_dict': self.model.state_dict(),
                    'opt_ae_state_dict': self.optimizer_ae.state_dict(),
                    'opt_disc_a_state_dict': self.opt_disc_a.state_dict(),
                    'opt_disc_b_state_dict': self.opt_disc_b.state_dict()
                }, os.path.join(os.getcwd(), self.save_path, 'settingc_n_{}.pt'.format(epoch)))
