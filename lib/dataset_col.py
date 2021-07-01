import torch
from torchvision import datasets, transforms
from matplotlib import pyplot as plt
import cv2
import numpy as np
import random
import scipy.ndimage as ndi
from tqdm import tqdm
import os
from PIL import Image
from skimage.io import imread
pad_size=1024
desired_size=512
class Dataset(torch.utils.data.Dataset):
    def __init__(self, img_paths, label_paths, transform=None,fusion=False):
        self.img_paths = img_paths
        self.label_paths = label_paths
        self.transform = transform[0]
        self.label_transform = transform[1]

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        img = cv2.imread(img_path)
        if self.label_paths is not None:
            label_path =self.label_paths[index]
            label = cv2.imread(label_path,cv2.IMREAD_GRAYSCALE) 
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (desired_size, desired_size))
        # img = imread(img_path)
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        if self.label_paths is not None:  
            label = cv2.resize(label,
                          (desired_size, desired_size))
            _, label = cv2.threshold(label, 127, 255, cv2.THRESH_BINARY)
            label = Image.fromarray(label)      
            if self.transform is not None:
                 label = self.label_transform(label)
        return img,img,label


    def __len__(self):
        return len(self.img_paths)
