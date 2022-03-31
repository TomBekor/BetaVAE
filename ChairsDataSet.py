import os
import torch
import numpy as np
import torch
from torch.utils.data import Dataset
from skimage import io
from os import path
import matplotlib.pyplot as plt

class ChairsDataSet(Dataset):
    
    def __init__(self, root_path = 'datasets/rendered_chairs', transform=None, rename=False):
        """
        Args:
            img_dir_path (string): Path to the directory with all the images.
            transform (callable, optional): Optional transform to be applied
                                            on a sample.
        """
        self.transform = transform
        self.root_path = root_path
        self.img_dir_dirs = list(os.listdir(self.root_path))
        self.img_paths = []
        self.labels = []
        for d in self.img_dir_dirs:
            if d == 'all_chair_names.mat': continue
            base = os.listdir(os.path.join(self.root_path, d, 'renders'))
            for n in base:
                try:
                    self.img_paths.append(os.path.join(self.root_path, d, 'renders', n))
                except:
                    continue

        
        

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img = io.imread(self.img_paths[idx])
        
        if self.transform:
            img = self.transform(img)

        return img, 1
    
