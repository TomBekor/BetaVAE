import os
import torch
import numpy as np
import torch
from torch.utils.data import Dataset
from skimage import io

class ButterflyDataSet(Dataset):
    
    def __init__(self, root_path, transform=None, rename=False):
        """
        Args:
            img_dir_path (string): Path to the directory with all the images.
            transform (callable, optional): Optional transform to be applied
                                            on a sample.
        """
        self.transform = transform
        self.root_path = root_path
        self.img_dir_path = os.path.join(self.root_path, 'images')
        # save image names:
        self.img_names = os.listdir(self.img_dir_path)
        

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = os.path.join(self.img_dir_path,
                                self.img_names[idx])
        y = self.img_names[idx][2: 4]
        if y == '01':
            y = 10
        else:
            y = int(y[1])

        img = io.imread(img_path)
        if self.transform:
            img = self.transform(img)

        return img, y 