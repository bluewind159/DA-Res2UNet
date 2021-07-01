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
from misc_functions import *
pad_size=1024
desired_size=512
import random
import cv2
import os
import datetime
from pengzhang import Morphology_Erode,Morphology_Dilate
from mpl_toolkits.mplot3d import Axes3D
def change_size(image,label):
    acc=[]
    for i in range(image.shape[0]):
        if image[i].sum()==0:
            acc.append(i)
    image=np.delete(image,acc,axis=0)
    label=np.delete(label,acc,axis=0)
    acc=[]
    for i in range(image.shape[1]):
        if image[:,i,:].sum()==0:
            acc.append(i)
    image=np.delete(image,acc,axis=1)
    label=np.delete(label,acc,axis=1)
    return image,label    
    
class Dataset(torch.utils.data.Dataset):
    def __init__(self, img_paths, label_paths=None, transform=None, fusion=False):
        self.img_paths = img_paths
        self.label_paths = label_paths
        self.transform = transform[0]
        self.label_transform = transform[1]
        self.fusion=fusion
    
    def gamma_trans(self,img0,gamma):
        gamma_table = [np.power(x/255.0,gamma)*255.0 for x in range(256)]
        gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)
        return cv2.LUT(img0,gamma_table)
        
    def Z_ScoreNormalization(self,x):
        mu=np.average(x)
        sigma=np.std(x)
        x = (x - mu) / sigma
        return x
    def MaxMinNormalization(self,x):
        Max=np.max(x)
        Min=np.min(x)
        x = (x - Min) / (Max - Min)
        return x
    def __getitem__(self, index):
        img_path = self.img_paths[index]
        im = cv2.imread(img_path)
        if self.label_paths is not None:
            label_path =self.label_paths[index]
            label = cv2.imread(label_path,cv2.IMREAD_GRAYSCALE)
        #im,label=change_size(im,label)
        im_gamma=self.gamma_trans(im,0.6)
        im_gray_gamma=cv2.cvtColor(im_gamma,cv2.COLOR_BGR2GRAY)
        #im_gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
        clahe=cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
        dst_gamma=clahe.apply(im_gray_gamma)
        #dst=clahe.apply(im_gray)

        im = cv2.resize(dst_gamma, (desired_size, desired_size))
        im = self.Z_ScoreNormalization(im)
        im = Image.fromarray(im)
        if self.transform is not None:
            im = self.transform(im)
        if self.label_paths is not None:  
            label = cv2.resize(label,
                          (desired_size, desired_size))
            _, label = cv2.threshold(label, 127, 255, cv2.THRESH_BINARY)
            label = Image.fromarray(label)      
            if self.transform is not None:
                 label = self.label_transform(label)
        im_new=im
        
        
        '''    
        unnormalize = NormalizeInverse([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        image233 = unnormalize(im_new.cpu())
        image233 = image233.data.cpu().numpy()
        image233 = np.uint8(image233 * 255).transpose(1,2,0)
        image233 = cv2.resize(image233, (1024, 1024))
        #image233 = cv2.cvtColor(image233, cv2.COLOR_BGR2RGB)
        cv2.imshow('image',image233)
        cv2.waitKey(0)
        image233 = unnormalize(im_acc1.cpu())
        image233 = image233.data.cpu().numpy()
        image233 = np.uint8(image233 * 255).transpose(1,2,0)
        image233 = cv2.resize(image233, (1024, 1024))
        #image233 = cv2.cvtColor(image233, cv2.COLOR_BGR2RGB)
        cv2.imshow('image',image233)
        cv2.waitKey(0)
        image233 = unnormalize(im_acc2.cpu())
        image233 = image233.data.cpu().numpy()
        image233 = np.uint8(image233 * 255).transpose(1,2,0)
        image233 = cv2.resize(image233, (1024, 1024))
        #image233 = cv2.cvtColor(image233, cv2.COLOR_BGR2RGB)
        cv2.imshow('image',image233)
        cv2.waitKey(0)
        image233 = unnormalize(acc.cpu())
        image233 = image233.data.cpu().numpy()
        image233 = np.uint8(image233 * 255).transpose(1,2,0)
        image233 = cv2.resize(image233, (1024, 1024))
        #image233 = cv2.cvtColor(image233, cv2.COLOR_BGR2RGB)
        cv2.imshow('image',image233)
        cv2.waitKey(0)
        assert 0==1
        '''
        return im,im_new, label

    def __len__(self):
        return len(self.img_paths)
