# -*- coding: utf-8 -*-
"""
Created on Fri Aug  5 16:09:25 2022

@author: tnsak
"""
import os
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Resize, ToTensor, ToPILImage
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class BWDataset(Dataset):
    
    def __init__(self, label_csv, data_dir='.', transform=Compose([ToPILImage(), Resize([224, 224]), ToTensor()])):
        '''
        Constructor for data loader, just loading in from file locs
        Also need to specify base dir of file paths since different
        from when i created them in R. 
        Not sure if this should go in a config file or if its okay 
        as a param in the data loade
        '''
        df = pd.read_csv(label_csv)
        # some files are NA bc I exported all w/o filtering, drop them now
        df = df[-np.isnan(df.wigMax)]
        self.file = [os.path.join(data_dir, x) for x in list(df.file)]
        self.csvname = label_csv
        self.species = df.species.tolist()
        self.transform = transform
    
    def __len__(self):
        return len(self.file)
    
    def __getitem__(self, ix):
        image = np.load(self.file[ix])
        # just repeating to fake RGB?
        image = np.repeat(image[..., np.newaxis], 3, -1)
        print(image.shape)
        label = self.species[ix]
        # make tensor
        if(self.transform):
            image = self.transform(image)
        print(image.shape)
        '''
        very unclear how I need to modify image input from uint8 nparray
        do lables need to be one-hot encoding??
        Models are expecting RGB? But I have gray?
        Some SE answers say just repeat the channel 3 times
        '''
        return image, label
    
    def showImg(self, ix):
        image, label = self[ix]
        image = np.moveaxis(image.numpy()*255, 0, -1).astype(np.uint8)[:,:,0]
        plt.imshow(np.flip(image))
        plt.title('Species: ' + label + ' File: ' + os.path.basename(self.file[ix])) 
        plt.yticks(np.linspace(0, 128, 5), np.linspace(96, 0, 5))
        plt.ylabel('Frequency (kHz)')
        plt.show()
        
    def multiImg(self, ix, ncol=5):
        nrow = int(np.ceil(len(ix)/ncol))
        fig, ax = plt.subplots(nrow, ncol)
        # fig = plt.figure()
        # gs = fig.add_gridspec(int(np.ceil(len(ix)/ncol)), ncol, hspace=0, wspace=0)
        # ax = gs.subplots(sharex=True, sharey=True)
        for i, v in enumerate(ix):
            im, lab = self[v]
            im = np.moveaxis(im.numpy()*255, 0, -1).astype(np.uint8)[:,:,0]
            im = np.flip(im)
            row = int(np.floor(i / ncol))
            col = i % ncol
            # print('Row ' + str(row) + ' Col ' + str(col))
            ax[row, col].imshow(im)
            ax[row, col].set_xticks([])
            ax[row, col].set_yticks([])
            # ax[row, col].margins(0.01, tight=True)
        # fig.tight_layout(pad=.05)
        fig.subplots_adjust(bottom=0, top=1, left=0, right=1)
        