import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from skimage import io
import math
from model import FCBetaVAE

class DisentanglementDataSet(Dataset):
    def __init__(self, dsprites_dataset, dataset_size=1000, L=64, model_path='models/DSprites/beta_1/epoch_7.pth'):
        self.L = L
        self.dsprites_dataset = dsprites_dataset
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.generating_factors = {
            0: ('Shape', np.array([0,1,2])),
            1: ('Scale', np.linspace(0.5,1,6)),
            2: ('Orientation', np.linspace(0,2*math.pi,40)),
            3: ('Position X', np.linspace(0,1,32)),
            4: ('Position Y', np.linspace(0,1,32))
        }

        self.model = FCBetaVAE(latent_dim=10, input_dim=4096).to(self.device)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

        self.z_b_diff_lst = []
        for _ in range(dataset_size):
            t, y = self.generate_z_b_diff()
            self.z_b_diff_lst.append((np.array(t.detach().cpu()), y))
        

    def generate_z_b_diff(self):
        fixed_y = np.random.choice(list(self.generating_factors.keys()))
        fixed_y_latent = np.random.choice(self.generating_factors[fixed_y][1])
        generated_imgs = self.dsprites_dataset.images_from_data_gen(sample_size=self.L, y=fixed_y, y_lat=fixed_y_latent)

        generated_imgs = generated_imgs.to(self.device)
        latents, _, _ = self.model.encode(generated_imgs)

        z1_lst = latents[:int(self.L/2)]
        z2_lst = latents[int(self.L/2):]

        z_l_diff_lst = torch.abs(z1_lst - z2_lst)
        z_b_diff = (1/self.L) * torch.sum(z_l_diff_lst, dim=0)
        return z_b_diff, fixed_y

    def __len__(self):
        return len(self.z_b_diff_lst)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        arr, y = self.z_b_diff_lst[idx]
        return torch.Tensor(arr), y