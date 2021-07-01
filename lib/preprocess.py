import os
from glob import glob
import numpy as np
import cv2
from skimage import measure
import pandas as pd
from tqdm import tqdm


def scale_radius(src, img_size, padding=False):
    x = src[src.shape[0] // 2, ...].sum(axis=1)
    r = (x > x.mean() / 10).sum() // 2
    yx = src.sum(axis=2)
    region_props = measure.regionprops((yx > yx.mean() / 10).astype('uint8'))
    yc, xc = np.round(region_props[0].centroid).astype('int')
    x1 = max(xc - r, 0)
    x2 = min(xc + r, src.shape[1] - 1)
    y1 = max(yc - r, 0)
    y2 = min(yc + r, src.shape[0] - 1)
    dst = src[y1:y2, x1:x2]
    dst = cv2.resize(dst, dsize=None, fx=img_size/(2*r), fy=img_size/(2*r))
    if padding:
        pad_x = (img_size - dst.shape[1]) // 2
        pad_y = (img_size - dst.shape[0]) // 2
        dst = np.pad(dst, ((pad_y, pad_y), (pad_x, pad_x), (0, 0)), 'constant')
    return dst


def normalize(src, img_size):
    dst = cv2.addWeighted(src, 4, cv2.GaussianBlur(src, (0, 0), img_size / 30), -4, 128)
    return dst


def remove_boundaries(src, img_size):
    mask = np.zeros(src.shape)
    cv2.circle(
        mask,
        center=(src.shape[1] // 2, src.shape[0] // 2),
        radius=int(img_size / 2 * 0.9),
        color=(1, 1, 1),
        thickness=-1)
    dst = src * mask + 128 * (1 - mask)
    return dst


def preprocess(dataset, img_size, scale=False, norm=False, pad=False, remove=False):
    data_location = ''
    if dataset == 'CHASE':
        training_images_loc = data_location + 'CHASE/train/image/'
        training_label_loc = data_location + 'CHASE/train/label/'
        validate_images_loc = data_location + 'CHASE/validate/images/'
        validate_label_loc = data_location + 'CHASE/validate/labels/'
        train_files = os.listdir(training_images_loc)
        validate_files = os.listdir(validate_images_loc)
        img_train_paths = training_images_loc + np.array(train_files,dtype='object')
        label_train_paths = training_label_loc + np.array([i.split('_')[0]+"_"+i.split('_')[1].split(".")[0] +"_1stHO.png" for i in train_files],dtype='object')
        img_val_paths = validate_images_loc + np.array(validate_files,dtype='object')
        label_val_paths = validate_label_loc + np.array([i.split('_')[0]+"_"+i.split('_')[1].split(".")[0] +"_1stHO.png" for i in validate_files],dtype='object')
    
    elif dataset == 'DRIVE':
        training_images_loc = data_location + 'DRIVE/train/image/'
        training_label_loc = data_location + 'DRIVE/train/label/'
        validate_images_loc = data_location + 'DRIVE/validate/images/'
        validate_label_loc = data_location + 'DRIVE/validate/labels/'
        train_files = os.listdir(training_images_loc)
        validate_files = os.listdir(validate_images_loc)
        img_train_paths = training_images_loc + np.array(train_files,dtype='object')
        label_train_paths = training_label_loc + np.array([i.split('_')[0]+"_"+i.split('_')[1].split(".")[0] +"_1stHO.png" for i in train_files],dtype='object')
        img_val_paths = validate_images_loc + np.array(validate_files,dtype='object')
        label_val_paths = validate_label_loc + np.array([i.split('_')[0]+"_"+i.split('_')[1].split(".")[0] +"_1stHO.png" for i in validate_files],dtype='object')
    else:
        NotImplementedError

    dir_name = 'processed/%s/images_%d' %(dataset, img_size)
    if scale:
        dir_name += '_scaled'
    if norm:
        dir_name += '_normed'
    if pad:
        dir_name += '_pad'
    if remove:
        dir_name += '_rm'

    os.makedirs(dir_name, exist_ok=True)
    for i in tqdm(range(len(img_paths))):
        img_path = img_paths[i]
        label_path = label_paths[i]
        print(img_path)
        print(label_path)
        if os.path.exists(os.path.join(dir_name, os.path.basename(img_path))):
            continue
        img = cv2.imread(img_path)
        label=cv2.imread(label_path,cv2.IMREAD_GRAYSCALE)
        cv2.imshow('label',label)
        cv2.waitKey (0)
        pad_size=1024
        old_size = img.shape[:2]  # old_size is in (height, width) format
        delta_w = pad_size - old_size[1]
        delta_h = pad_size - old_size[0]
    
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)
    
        color = [0, 0, 0]
        color2 = [0]
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                    value=color)
    
        label = cv2.copyMakeBorder(label, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                       value=color2)
        try:
            if scale:
                img = scale_radius(img, img_size=img_size, padding=pad)
                label = scale_radius(label, img_size=img_size, padding=pad)
        except Exception as e:
            print(img_paths[i])
            print(label_paths[i])
    
        img = cv2.resize(img, (img_size, img_size))
        labels = cv2.resize(label,
                          (img_size, img_size))
        if norm:
            img = normalize(img, img_size=img_size)
            label = normalize(label, img_size=img_size)
        if remove:
            img = remove_boundaries(img, img_size=img_size)
            label = remove_boundaries(label, img_size=img_size)
        cv2.imshow('img',img)
        cv2.waitKey (0)
        cv2.imshow('label',label)
        cv2.waitKey (0)
        assert 0==1
        cv2.imwrite(os.path.join(dir_name+'/img', os.path.basename(img_path)), img)
        cv2.imwrite(os.path.join(dir_name+'/label', os.path.basename(label_path)), label)
    return dir_name+'/img', dir_name+'/label'
