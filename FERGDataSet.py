import os
import torch
import numpy as np
import torch
from torch.utils.data import Dataset
from skimage import io
import os
from os import listdir
from PIL import Image

class FERGDataSet(Dataset):
    
    def __init__(self, root_path='datasets/FERG_DB_256', transform=None, rename=False):
        """
        Args:
            img_dir_path (string): Path to the directory with all the images.
            transform (callable, optional): Optional transform to be applied
                                            on a sample.
        """
        self.root_path = root_path
        self.figures_name = os.listdir(self.root_path)
        figs = []
        for f in self.figures_name:
            if f.islower():
                figs.append(f)
        self.figures_name = figs
        names_to_number = {'aia':0, 'bonnie':1, 'jules':2,
         'malcolm':3, 'mery':4, 'ray':5}
        emotions_to_number = {'anger':0,'disgust':1,'fear':2,
        'joy':3,'netural':4,'sadness':5,'surpise':6}
        self.imgs_path = []
        self.figure_labels = []
        self.emotions_labels = []
        for figure in self.figures_name:
            emotions = os.listdir(os.path.join(self.root_path, figure))
            lst = []
            for f in emotions:
                if f.islower():
                    lst.append(f)
            emotions = lst
            for e in emotions:
                imgs = os.listdir(os.path.join(self.root_path, figure, e))
            
                for i in imgs:
                    path = os.path.join(self.root_path, figure, e, i)
                    try: 
                        img = io.imread(path, pilmode='RGB')
                        self.imgs_path.append(path)
                        self.emotions_labels.append(emotions_to_number[e])
                        self.figure_labels.append(names_to_number[figure])
                    except:
                        continue
        for filename in self.imgs_path:
            try:
                img = io.imread(path, pilmode='RGB')
            except (IOError, SyntaxError) as e:
                print('Bad file:', filename)

        self.transform = transform

        # save image names:
        

    def __len__(self):
        return len(self.imgs_path)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        x_path = self.imgs_path[idx]
        # y = self.figure_labels[idx]
        img = io.imread(x_path, pilmode='RGB')
        if self.transform:
            img = self.transform(img)
        img = torch.Tensor(img)
        return img, 1