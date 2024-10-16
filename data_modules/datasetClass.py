import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
import torchvision.transforms as T
from PIL import Image
import os
import numpy as np

class myDataset(Dataset):
    print("MESSAGE FROM DATALOADER:\nPastikan untuk menginisiasi folder Anda terlebih dahulu yaa. Dalam folder tersebut, harus ada dua macam folder: \n-> pseudoPaired\n-> unpairedData\nMasing-masing berisi dua subfolder bernama trainA dan trainB")
    def __init__(self, root, resize=256):
        assert root is not None, "Hayoo, tentukan dulu folder root Anda apa :)"
        self.paired_dir = os.path.join(root, "pseudoPaired")
        self.unpaired_dir = os.path.join(root, "unpairedData")

        self.paired_data_xp = sorted(os.listdir(os.path.join(self.paired_dir, "trainA")))
        self.paired_data_yp = sorted(os.listdir(os.path.join(self.paired_dir, "trainB")))
        self.unpaired_data_x = sorted(os.listdir(os.path.join(self.unpaired_dir, "trainA")))
        self.unpaired_data_y = sorted(os.listdir(os.path.join(self.unpaired_dir, "trainB")))
        self.transforms = T.Compose([T.Resize((resize, resize), Image.BICUBIC), 
                                              T.ToTensor(), 
                                              T.Normalize(mean=[0.5, 0.5, 0.5], 
                                                          std=[0.5, 0.5, 0.5])
                                              ])

    def get_preview_data(self):
        print(self.paired_data_xp[:10])
        print(self.paired_data_yp[:10])
        print(self.unpaired_data_x[:10])
        print(self.unpaired_data_y[:10])

    def get_len_data(self):
        return {"n_xp": len(self.paired_data_xp), "n_yp": len(self.paired_data_yp), "n_x": len(self.unpaired_data_x), "n_y": len(self.unpaired_data_y)}

    def flush_data(self):
        confirm = input("THIS IS A DESTRUCTIVE ACTION. ARE YOU SURE YOU WANT TO DELETE ALL OF THE DATA? [Y/N]")
        if confirm == "Y":
            for x in self.paired_data_xp:
                os.remove(
                    os.path.join(
                        os.path.join(self.paired_dir, "trainA"), x)
                ) 
            for x in self.paired_data_yp:
                os.remove(
                    os.path.join(
                        os.path.join(self.paired_dir, "trainB"), x)
                ) 
            for x in self.unpaired_data_x:
                os.remove(
                    os.path.join(
                        os.path.join(self.unpaired_dir, "trainA"), x)
                ) 
            for x in self.unpaired_data_y:
                os.remove(
                    os.path.join(
                        os.path.join(self.unpaired_dir, "trainB"), x)
                ) 
            self.paired_data_xp = []
            self.paired_data_yp = []
            self.unpaired_data_x = []
            self.unpaired_data_y = []
        else:
            return

    def __getitem__(self, idx, styleref_id=None):
        random_X_index = np.random.randint(0, len(self.paired_data_xp))
        random_Y_index = np.random.randint(0, len(self.unpaired_data_y))
        X_p = Image.open(os.path.join(
                        os.path.join(self.paired_dir, "trainA"), self.paired_data_xp[idx])).convert('RGB')
        Y_p = Image.open(os.path.join(
                        os.path.join(self.paired_dir, "trainB"), self.paired_data_yp[idx])).convert('RGB')
        X = Image.open(os.path.join(
                        os.path.join(self.paired_dir, "trainA"), self.paired_data_xp[random_X_index])).convert('RGB')
        Y = Image.open(os.path.join(
                        os.path.join(self.unpaired_dir, "trainB"), self.unpaired_data_y[random_Y_index])).convert('RGB')

        X_p = self.transforms(X_p)
        Y_p = self.transforms(Y_p)
        X = self.transforms(X)
        Y = self.transforms(Y)

        return {"X_p": X_p, "Y_p": Y_p, "X": X, "Y": Y}

    def __len__(self):
        return len(self.paired_data_xp)
