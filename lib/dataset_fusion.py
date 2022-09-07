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
        if self.fusion:
           print('fusion the data')
        else:
           print('not fusion the data')
        print('FUSION DATASETS!!!!!!!!!!!!!!!!!!!!!')
        
    def gamma_trans(self,img0,gamma):
        gamma_table = [np.power(x/255.0,gamma)*255.0 for x in range(256)]
        gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)
        return cv2.LUT(img0,gamma_table)
    def Z_ScoreNormalization(self,x,mu,sigma):
        x = (x - mu) / sigma
        return x
    def MaxMinNormalization(self,x):
        Max=np.max(x)
        Min=np.min(x)
        x = (x - Min) / (Max - Min)
        return x

    def deal_image(self,im,contrast=False):
        im_gamma=self.gamma_trans(im,0.6)
        #im_gray_gamma=im[:,:,1]
        im_gray_gamma=cv2.cvtColor(im_gamma,cv2.COLOR_BGR2GRAY)
        #clahe=cv2.createCLAHE(clipLimit=3,tileGridSize=(10,10))
        clahe=cv2.createCLAHE(clipLimit=2,tileGridSize=(8,8))
        dst_gamma=clahe.apply(im_gray_gamma)
        if contrast:
           dst_gamma=dst_gamma*0.2
        im = cv2.resize(dst_gamma, (desired_size, desired_size))
        mu=np.average(im)
        sigma=np.std(im)
        im = self.Z_ScoreNormalization(im,mu,sigma)
        return im,mu,sigma
    def __getitem__(self, index):
        img_path = self.img_paths[index]
        im = cv2.imread(img_path)
        if self.label_paths is not None:
            label_path =self.label_paths[index]
            label = cv2.imread(label_path,cv2.IMREAD_GRAYSCALE)
        im,label=change_size(im,label)
        #im = cv2.resize(im, (desired_size, desired_size))
        fu_ind=1
        fu_size=fu_ind*32
        dst = cv2.blur(im, (fu_size, fu_size))
        dst1 = cv2.convertScaleAbs(im,alpha=0.2,beta=100)
        dst2 = cv2.convertScaleAbs(im,alpha=1,beta=160)
        dst2 = cv2.convertScaleAbs(dst2,alpha=1,beta=-160)
        im,mu,sigma = self.deal_image(im)
        ##################################################################
        #im = cv2.resize(im, (desired_size, desired_size))
        #####################################################################
        dst,mu_d,sigma_d = self.deal_image(dst)
        dst1,mu_d1,sigma_d1 = self.deal_image(dst1)
        dst2,mu_d2,sigma_d2 = self.deal_image(dst2)
        im = Image.fromarray(im)
        dst = Image.fromarray(dst)
        dst1 = Image.fromarray(dst1)
        dst2 = Image.fromarray(dst2)
        
        if self.transform is not None:
            im = self.transform(im)
            dst = self.transform(dst)
            dst1 = self.transform(dst1)
            dst2 = self.transform(dst2)
        if self.label_paths is not None:  
            label = cv2.resize(label,
                          (desired_size, desired_size))
            _, label = cv2.threshold(label, 127, 255, cv2.THRESH_BINARY)
            #acc=Morphology_Dilate(label,Dil_time=2)
            #cv2.imwrite(label_path.replace('label','d_label'),acc)
            acc = cv2.imread(label_path.replace('label','d_label'),cv2.IMREAD_GRAYSCALE)
            _, acc = cv2.threshold(acc, 127, 255, cv2.THRESH_BINARY)
            label = Image.fromarray(label)
            acc = Image.fromarray(acc)
            if self.transform is not None:
                 label = self.label_transform(label)
                 acc = self.label_transform(acc)
        qqqqq=random.random()
        #print(qqqqq)
        im_fu1=dst1#*mask
        im_fu2=dst2#*mask
        #qqqqq=0.9
        #qsc=random.random()
        #if qsc>0.85:
           #qqqqq = 0.6
        #else:
           #qqqqq = 0
        #print(im.shape)
        if self.fusion:
            if qqqqq>0.85:
                #print('fusion the data')
                mask=(acc>0).long()
                anti_mask=(~(acc>0)).long()
                im_fu=dst*mask
                im_cl=im*mask
                im_acc1=torch.zeros(im.shape)
                im_acc2=torch.zeros(im.shape)
                block_size=32
                assert desired_size%block_size==0
                for i in range(int(desired_size/block_size)):
                    for j in range(int(desired_size/block_size)):
                        acc=random.random()
                        if  acc<=0.25:
                            im_acc1[i*block_size:(i+1)*block_size,j*block_size:(j+1)*block_size]=im_fu[i*block_size:(i+1)*block_size,j*block_size:(j+1)*block_size]
                        else:
                            im_acc2[i*block_size:(i+1)*block_size,j*block_size:(j+1)*block_size]=im_cl[i*block_size:(i+1)*block_size,j*block_size:(j+1)*block_size]
                            #im_acc=im_cl
                im_qwe=im*anti_mask
                im_new=im_acc1+im_acc2+im_qwe    
            elif qqqqq>0.7 and qqqqq<=0.85:
                im_new=im_fu1
                #print('duibidu')
            elif qqqqq>0.55 and qqqqq<=0.7:
                im_new=im_fu2
                #print('explore')
            else:
                im_new=im
                #print('origin')
        else:
            im_new=im
            
        
            
        return im,im_new, label,acc

    def __len__(self):
        return len(self.img_paths)
