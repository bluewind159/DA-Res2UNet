import time
import os
import math
import argparse
from glob import glob
from collections import OrderedDict
import random
import warnings
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import joblib

from sklearn.model_selection import StratifiedKFold, train_test_split
from skimage.io import imread

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import datasets, models, transforms

from lib.dataset_col import Dataset
from lib.models.model_factory import get_model
from lib.utils import *
from lib.metrics import *
from lib.losses import *
from lib.optimizers import *
from lib.preprocess import preprocess
from lib.models.res2unet import Res2UNet
from lightweight_gan.lightweight_gan import LightweightGAN,Trainer
from lightweight_gan.cli import build_model
import cv2
from combine_model import combine_model
from misc_functions import *

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default='combine_model',
                        help='model name: (default: arch+timestamp)')
    parser.add_argument('--arch', '-a', metavar='ARCH', default='efficientnet',
                        help='model architecture: ' +
                        ' (default: resnet34)')
    parser.add_argument('--dropout_p', default=0, type=float)
    parser.add_argument('--loss', default='BCELoss',
                        choices=['CrossEntropyLoss', 'FocalLoss', 'MSELoss', 'multitask','BCELoss'])
    parser.add_argument('--epochs', default=1000, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch_size', default=1, type=int,
                        metavar='N', help='mini-batch size (default: 32)')
    parser.add_argument('--img_size', default=512, type=int,
                        help='input image size (default: 288)')
    parser.add_argument('--optimizer', default='Adam')
    parser.add_argument('--pred_type', default='classification',
                        choices=['classification', 'regression', 'multitask'])
    parser.add_argument('--scheduler', default='CosineAnnealingLR',
                        choices=['CosineAnnealingLR', 'ReduceLROnPlateau'])
    parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--min_lr', default=1e-6, type=float,
                        help='minimum learning rate')
    parser.add_argument('--weight-decay', default=1e-4, type=float,
                        help='weight decay')
    # dataset
    parser.add_argument('--class_aware', default=False, type=str2bool)
    parser.add_argument('--train_dataset',
                        default='CHASE')
    parser.add_argument('--pretrained_model')
    parser.add_argument('--pseudo_labels')
    args = parser.parse_args()

    return args

def train(args, train_dataset, model, criterion, optimizer, epoch):
    losses = AverageMeter()
    scores = AverageMeter()
    specificity = AverageMeter()
    sensitivity = AverageMeter()
    model_g,model_ext=model
    model_g.train()
    model_ext.eval()
    transform233 = transforms.Compose([
                        #transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                    ])
    unnormalize = NormalizeInverse([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    #for i, (input, target) in tqdm(enumerate(train_loader), total=len(train_loader)):
    for i in tqdm(range(1000),total=1000):
        input, input_new,target=train_dataset[0]
        input = input.cuda().unsqueeze(0)
        target = target.cuda().unsqueeze(0)
        G=model_g
        latents = torch.randn(1, 256).cuda()
        generated_images = G(latents)
        generated_images = generated_images.clamp_(0., 1.)
        generated_images = generated_images
        generated_images = transform233(generated_images)
        output = model_ext(generated_images)
        loss = criterion(output, target)
        # compute gradient and do optimizing step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        output=(output >0.5).long()
        score,Sp,Sen,_ = accuracy(output, target)
        losses.update(loss.item(), input.size(0))
        scores.update(score, input.size(0))
        specificity.update(Sp, input.size(0))
        sensitivity.update(Sen, input.size(0))
    label = output[0].data.cpu().numpy()
    label = np.uint8(label * 255).transpose(1,2,0)
    label = cv2.resize(label, (1024, 1024))
    image233 = unnormalize(generated_images[0].cpu())
    image233 = image233.data.cpu().numpy()
    image233 = np.uint8(image233 * 255).transpose(1,2,0)
    image233 = cv2.resize(image233, (1024, 1024))
    image233 = cv2.cvtColor(image233, cv2.COLOR_BGR2RGB)
    plt.subplot(1,2,1)
    plt.imshow(image233)
    plt.title('pic 1')
    plt.subplot(1,2,2)
    plt.imshow(label, cmap='gray')
    plt.title('pic 2')
    plt.savefig('./combine_image/CHASE/combine_'+str(epoch)+'.png')
    return losses.avg, scores.avg, specificity.avg, sensitivity.avg


def validate(args, val_loader, model, criterion):
    losses = AverageMeter()
    scores = AverageMeter()
    specificity = AverageMeter()
    sensitivity = AverageMeter()
    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for i, (input, target) in tqdm(enumerate(val_loader), total=len(val_loader)):
            input = input.cuda()
            target = target.cuda()
            output = model(input)
            loss = criterion(output, target)
            output=(output >0.5).long()
            score,Sp,Sen = accuracy(output, target)
            losses.update(loss.item(), input.size(0))
            scores.update(score, input.size(0))
            specificity.update(Sp.item(), input.size(0))
            sensitivity.update(Sen, input.size(0))
    return losses.avg, scores.avg, specificity.avg, sensitivity.avg


def main():
    args = parse_args()
    if args.name is None:
        args.name = '%s_%s' % (args.arch, datetime.now().strftime('%m%d%H'))

    if not os.path.exists('models/%s' % args.name):
        os.makedirs('models/%s' % args.name)

    print('Config -----')
    for arg in vars(args):
        print('- %s: %s' % (arg, getattr(args, arg)))
    print('------------')

    with open('models/%s/args.txt' % args.name, 'w') as f:
        for arg in vars(args):
            print('- %s: %s' % (arg, getattr(args, arg)), file=f)

    joblib.dump(args, 'models/%s/args.pkl' % args.name)

    if args.loss == 'CrossEntropyLoss':
        criterion = nn.CrossEntropyLoss().cuda()
    elif args.loss == 'FocalLoss':
        criterion = FocalLoss().cuda()
    elif args.loss == 'MSELoss':
        criterion = nn.MSELoss().cuda()
    elif args.loss == 'multitask':
        criterion = {
            'classification': nn.CrossEntropyLoss().cuda(),
            'regression': nn.MSELoss().cuda(),
        }
    elif args.loss == 'BCELoss':
        criterion = nn.BCELoss().cuda()
    else:
        raise NotImplementedError

    if args.pred_type == 'classification':
        num_outputs = 5
    elif args.pred_type == 'regression':
        num_outputs = 1
    elif args.loss == 'multitask':
        num_outputs = 6
    else:
        raise NotImplementedError

    cudnn.benchmark = True

    train_transform = []
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    train_label_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_label_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # data loading code
    data_location=''
    if 'CHASE' in args.train_dataset:
        training_images_loc = data_location + 'CHASE/train/image/'
        training_label_loc = data_location + 'CHASE/train/label/'
        validate_images_loc = data_location + 'CHASE/validate/images/'
        validate_label_loc = data_location + 'CHASE/validate/labels/'
        test_images_loc = data_location + 'CHASE/test/image/'
        test_label_loc = data_location + 'CHASE/test/label/'
        train_files = os.listdir(training_images_loc)
        validate_files = os.listdir(validate_images_loc)
        test_files = os.listdir(test_images_loc)
        train_img_paths = training_images_loc + np.array(train_files,dtype='object')
        train_label_paths = training_label_loc + np.array([i.split('_')[0]+"_"+i.split('_')[1].split(".")[0] +"_1stHO.png" for i in train_files],dtype='object')
        val_img_paths = validate_images_loc + np.array(validate_files,dtype='object')
        val_label_paths = validate_label_loc + np.array([i.split('_')[0]+"_"+i.split('_')[1].split(".")[0] +"_1stHO.png" for i in validate_files],dtype='object')
        test_img_paths = test_images_loc + np.array(test_files,dtype='object')
        test_label_paths = test_label_loc + np.array([i.split('_')[0]+"_"+i.split('_')[1].split(".")[0] +"_1stHO.png" for i in test_files],dtype='object')
   
    if 'DRIVE' in args.train_dataset:
        training_images_loc = data_location + 'DRIVE/train/images/'
        training_label_loc = data_location + 'DRIVE/train/labels/'
        validate_images_loc = data_location + 'DRIVE/validate/images/'
        validate_label_loc = data_location + 'DRIVE/validate/labels/'
        test_images_loc = data_location + 'DRIVE/test/images/'
        test_label_loc = data_location + 'DRIVE/test/labels/'
        train_files = os.listdir(training_images_loc)
        validate_files = os.listdir(validate_images_loc)
        test_files = os.listdir(test_images_loc)
        train_img_paths = training_images_loc + np.array(train_files,dtype='object')
        train_label_paths = training_label_loc + np.array([i.split('_')[0]+"_manual1.png" for i in train_files],dtype='object')
        val_img_paths = validate_images_loc + np.array(validate_files,dtype='object')
        val_label_paths = validate_label_loc + np.array([i.split('_')[0]+"_manual1.png" for i in validate_files],dtype='object')
        test_img_paths = test_images_loc + np.array(test_files,dtype='object')
        test_label_paths = test_label_loc + np.array([i.split('_')[0]+"_manual1.png" for i in test_files],dtype='object')
    
    if 'gen' in args.train_dataset:
        training_images_loc = data_location + 'generate_data/resized_train/image/'
        validate_images_loc = data_location + 'generate_data/resized_test/images/'
        train_files = os.listdir(training_images_loc)
        validate_files = os.listdir(validate_images_loc)
        train_img_paths = training_images_loc + np.array(train_files,dtype='object')
        val_img_paths = validate_images_loc + np.array(validate_files,dtype='object')
    
    best_losses = []
    best_scores = []
    
    # train
    train_set = Dataset(
        train_img_paths,
        train_label_paths,
        transform=[train_transform,train_label_transform])
    class_sample_counts = 2
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=False if args.class_aware else True,
        num_workers=4,
        sampler=sampler if args.class_aware else None)

    val_set = Dataset(
        val_img_paths,
        val_label_paths,
        transform=[val_transform,val_label_transform])
        
    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4)

    model_ext = Res2UNet(3,1)
    model_ext = model_ext.cuda()
    model_g = build_model()
    model_g = model_g.GAN.G.cuda()
    
    
    if args.pretrained_model is not None:
            model_ext.load_state_dict(torch.load('models/%s/model.pth' % (args.pretrained_model)))
            print('load_model_ext')
    #model=combine_model(model_g,model_ext)
    #model(1,256)
            
    if args.optimizer == 'Adam':
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model_g.parameters()), lr=args.lr)
    elif args.optimizer == 'AdamW':
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model_g.parameters()), lr=args.lr)
    elif args.optimizer == 'RAdam':
        optimizer = RAdam(
            filter(lambda p: p.requires_grad, model_g.parameters()), lr=args.lr)
    elif args.optimizer == 'SGD':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model_g.parameters()), lr=args.lr,
                              momentum=args.momentum, weight_decay=args.weight_decay, nesterov=args.nesterov)

    if args.scheduler == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=args.min_lr)
    elif args.scheduler == 'ReduceLROnPlateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=args.factor, patience=args.patience,
                                                   verbose=1, min_lr=args.min_lr)

    log = pd.DataFrame(index=[], columns=[
        'epoch', 'loss', 'score', 'val_loss', 'val_score'
    ])
    log = {
        'epoch': [],
        'loss': [],
        'score': [],
        'val_loss': [],
        'val_score': [],
    }

    best_loss = float('inf')
    best_score = 0
    for epoch in range(args.epochs):
        print('Epoch [%d/%d]' % (epoch + 1, args.epochs))

        # train for one epoch
        train_loss, train_score, train_sp, train_sen = train(
            args, train_set, (model_g,model_ext), criterion, optimizer, epoch)
        # evaluate on validation set
        #val_loss, val_score, val_sp, val_sen = validate(args, val_set, (model_g,model_ext), criterion)

        if args.scheduler == 'CosineAnnealingLR':
            scheduler.step()
        elif args.scheduler == 'ReduceLROnPlateau':
            scheduler.step(val_loss)

        print('loss %.4f - score %.4f -Sp %.4f -Sen %.4f'
              % (train_loss, train_score, train_sp, train_sen))
        #print('val_loss %.4f - val_score %.4f - val_Sp %.4f - val_Sen %.4f'
        #      % (val_loss, val_score, val_sp, val_sen))
        val_loss=0
        val_score=0
        log['epoch'].append(epoch)
        log['loss'].append(train_loss)
        log['score'].append(train_score)
        log['val_loss'].append(val_loss)
        log['val_score'].append(val_score)

        pd.DataFrame(log).to_csv('models/%s/log.csv' % (args.name), index=False)

        #if train_score > best_score:
        #    torch.save(model_g.state_dict(), 'models/%s/model.pth' % (args.name))
        #    best_loss = train_loss
        #    best_score = train_score
        #    print("=> saved best model")

    print('val_loss:  %f' % best_loss)
    print('val_score: %f' % best_score)
    
    best_losses.append(best_loss)
    best_scores.append(best_score)

    results = pd.DataFrame({
        'best_loss': best_losses + [np.mean(best_losses)],
        'best_score': best_scores + [np.mean(best_scores)],
    })

    print(results)
    results.to_csv('models/%s/results.csv' % args.name, index=False)

    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
