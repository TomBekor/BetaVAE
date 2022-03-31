import os
import torch
import numpy as np
import torch
from torch.utils.data import Dataset
from skimage import io

class FlowerDataSet(Dataset):
    
    def __init__(self, root_path= 'datasets/flower_data/', transform=None, rename=False):
        """
        Args:
            img_dir_path (string): Path to the directory with all the images.
            transform (callable, optional): Optional transform to be applied
                                            on a sample.
        """
        self.transform = transform
        self.root_path = root_path
        dirs = list(os.listdir(self.root_path))
        self.img_paths = []
        self.labels = []
        for d in dirs:
            base = os.listdir(os.path.join(self.root_path, d)) 
            for n in base:
                imgs_dir = os.listdir(os.path.join(self.root_path, d, n))
                for img in imgs_dir:
                    self.labels.append(int(n))
                    self.img_paths.append(os.path.join(self.root_path, d, n, img))


    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img = io.imread(self.img_paths[idx])
        
        if self.transform:
            img = self.transform(img)

        return img, self.labels[idx]