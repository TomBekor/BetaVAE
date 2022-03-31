import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from skimage import io

class CelebADataSet(Dataset):
    def __init__(self, root_path, transform=None):
        """
        Args:
            img_dir_path (string): Path to the directory with all the images.
            transform (callable, optional): Optional transform to be applied
                                            on a sample.
        """
        self.transform = transform
        self.root_path = root_path
        self.img_dir_path = os.path.join(self.root_path, 'img_align_celeba')
        self.att_path = os.path.join(self.root_path, 'list_attr_celeba.csv')

        # save image names:
        self.img_names = os.listdir(self.img_dir_path)

        # save tags (gender):
        att_df = pd.read_csv(self.att_path)
        self.gender = att_df['Male'].values
        self.gender[self.gender==-1] = 0

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = os.path.join(self.img_dir_path,
                                self.img_names[idx])
        
        img = io.imread(img_path)
        if self.transform:
            img = self.transform(img)

        y = self.gender[idx]

        return img, y