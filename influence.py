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

from lib.dataset import Dataset
from lib.models.model_factory import get_model
from lib.utils import *
from lib.metrics import *
from lib.losses import *
from lib.optimizers import *
from lib.preprocess import preprocess
from lib.models.res2unet import Res2UNet
import cv2
from misc_functions import *
import random
from pengzhang import Morphology_Erode,Morphology_Dilate
from pytorch_influence_functions import calc_img_wise, calc_all_grad_then_test, calc_influence_double
import pytorch_influence_functions as ptif
from influence_utils import parallel
from influence_utils import nn_influence_utils


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default=None,
                        help='model name: (default: arch+timestamp)')
    parser.add_argument('--arch', '-a', metavar='ARCH', default='efficientnet',
                        help='model architecture: ' +
                        ' (default: resnet34)')
    parser.add_argument('--dropout_p', default=0, type=float)
    parser.add_argument('--loss', default='BCELoss',
                        choices=['CrossEntropyLoss', 'FocalLoss', 'MSELoss', 'multitask','BCELoss'])
    parser.add_argument('--epochs', default=500, type=int, metavar='N',
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
    parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--min_lr', default=1e-5, type=float,
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

                
def train(args, train_loader, model, criterion, optimizer, epoch, train_set):
    losses = AverageMeter()
    scores = AverageMeter()
    specificity = AverageMeter()
    sensitivity = AverageMeter()
    model.train()

    for i, (input,input_new, target) in tqdm(enumerate(train_loader), total=len(train_loader)):
        input = input.cuda()
        input_new=input_new.cuda()
        index=random.randint(0,len(train_loader)-1)
        afafaf=random.random()
        if afafaf>0.8:
            add_noise_mark=True
        else:
            add_noise_mark=False
        _,_,add_mark=train_set[index]
        #with torch.no_grad():
        #    add_mark=model(add_image.unsqueeze(0).cuda())
        #    add_mark=add_mark[0]
        block_size=16
        desired_size=input.size(-1)
        im_acc=torch.zeros(add_mark.size())
        assert desired_size%block_size==0
        for i in range(int(desired_size/block_size)):
            for j in range(int(desired_size/block_size)):
                acc=random.random()
                bblock = add_mark[:,i*block_size:(i+1)*block_size,j*block_size:(j+1)*block_size]
                if acc>0.3:
                    im_acc[:,i*block_size:(i+1)*block_size,j*block_size:(j+1)*block_size]=bblock*0
                else:
                    aaaa=random.random()
                    trans=random.random()
                    if trans>0.5:
                        bblock=bblock.transpose(2,1)
                    r=random.randint(0,15)
                    c=random.randint(0,15)
                    bblock=torch.roll(bblock,r,dims=1)
                    bblock=torch.roll(bblock,c,dims=2)
                    im_acc[:,i*block_size:(i+1)*block_size,j*block_size:(j+1)*block_size]=bblock*aaaa   
        if not torch.equal(add_mark,target):
            if add_noise_mark:
                im_acc=im_acc.cuda()
                target = target.cuda()
                mask=(target>0).long().cuda()
                anti_mask=(~(target>0)).long().cuda()
                #im_acc=im_acc*anti_mask
                input_new=input_new#+im_acc
        target = target.cuda()
        output = model(input)
        loss = criterion(output, target)
        # compute gradient and do optimizing step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        output=(output >0.4).long()
        score,Sp,Sen = accuracy(output, target)
        losses.update(loss.item(), input.size(0))
        scores.update(score, input.size(0))
        specificity.update(Sp, input.size(0))
        sensitivity.update(Sen, input.size(0))
        if False:
            output_new = model(input_new)
            output_new = (output_new >0.5).long()
            unnormalize = NormalizeInverse([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            image = unnormalize(input[0].cpu())
            image = image.data.cpu().numpy()
            image = np.uint8(image * 255).transpose(1,2,0)
            image = cv2.resize(image, (1024, 1024))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            ta = target[0].data.cpu().numpy()
            ta = np.uint8(ta * 255).transpose(1,2,0)
            ta = cv2.resize(ta, (1024, 1024))
            #image233 = cv2.cvtColor(image233, cv2.COLOR_BGR2RGB)
            origin = output[0].data.cpu().numpy()
            origin = np.uint8(origin * 255).transpose(1,2,0)
            origin = cv2.resize(origin, (1024, 1024))
            #image233 = cv2.cvtColor(image233, cv2.COLOR_BGR2RGB)
            fusion = output_new[0].data.cpu().numpy()
            fusion = np.uint8(fusion * 255).transpose(1,2,0)
            fusion = cv2.resize(fusion, (1024, 1024))
            #image233 = cv2.cvtColor(image233, cv2.COLOR_BGR2RGB)
            plt.subplot(1,4,1)
            plt.imshow(image)
            plt.subplot(1,4,2)
            plt.imshow(ta, cmap='gray')
            plt.subplot(1,4,3)
            plt.imshow(origin, cmap='gray')
            plt.subplot(1,4,4)
            plt.imshow(fusion, cmap='gray')
            plt.show()
            assert 0==1
    return losses.avg, scores.avg, specificity.avg, sensitivity.avg


def validate(args, val_loader, model, criterion):
    losses = AverageMeter()
    scores = AverageMeter()
    specificity = AverageMeter()
    sensitivity = AverageMeter()
    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for i, (input,input_new, target) in tqdm(enumerate(val_loader), total=len(val_loader)):
            input = input.cuda()
            target = target.cuda()
            output = model(input)
            loss = criterion(output, target)
            output=(output >0.4).long()
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

    if not os.path.exists('models/%s/%s' % (args.name,args.train_dataset)):
        os.makedirs('models/%s/%s' % (args.name,args.train_dataset))

    print('Config -----')
    for arg in vars(args):
        print('- %s: %s' % (arg, getattr(args, arg)))
    print('------------')

    with open('models/%s/%s/args.txt' % (args.name,args.train_dataset), 'w') as f:
        for arg in vars(args):
            print('- %s: %s' % (arg, getattr(args, arg)), file=f)

    joblib.dump(args, 'models/%s/%s/args.pkl' % (args.name,args.train_dataset) )

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
        train_files = os.listdir(training_images_loc)
        validate_files = os.listdir(validate_images_loc)
        train_img_paths = training_images_loc + np.array(train_files,dtype='object')
        train_label_paths = training_label_loc + np.array([i.split('_')[0]+"_"+i.split('_')[1].split(".")[0] +"_1stHO.png" for i in train_files],dtype='object')
        val_img_paths = validate_images_loc + np.array(validate_files,dtype='object')
        val_label_paths = validate_label_loc + np.array([i.split('_')[0]+"_"+i.split('_')[1].split(".")[0] +"_1stHO.png" for i in validate_files],dtype='object')
   
    if 'DRIVE' in args.train_dataset:
        training_images_loc = data_location + 'DRIVE/train/images/'
        training_label_loc = data_location + 'DRIVE/train/labels/'
        validate_images_loc = data_location + 'DRIVE/validate/images/'
        validate_label_loc = data_location + 'DRIVE/validate/labels/'
        train_files = os.listdir(training_images_loc)
        validate_files = os.listdir(validate_images_loc)
        train_img_paths = training_images_loc + np.array(train_files,dtype='object')
        train_label_paths = training_label_loc + np.array([i.split('_')[0]+"_manual1.png" for i in train_files],dtype='object')
        val_img_paths = validate_images_loc + np.array(validate_files,dtype='object')
        val_label_paths = validate_label_loc + np.array([i.split('_')[0]+"_manual1.png" for i in validate_files],dtype='object')
    
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
        transform=[train_transform,train_label_transform],
        fusion=True)
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

    model = Res2UNet(3,1)
    model = model.cuda()
    if args.pretrained_model is not None:
        model.load_state_dict(torch.load('models/%s/model.pth' % (args.pretrained_model)))

    if args.optimizer == 'Adam':
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    elif args.optimizer == 'AdamW':
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    elif args.optimizer == 'RAdam':
        optimizer = RAdam(
            filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    elif args.optimizer == 'SGD':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr,
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

    config_influence=ptif.get_default_config()
    config_influence['num_classes']=5
    config_influence['dataset']='CHASE'
    config_influence['damp']=5e-3
    config_influence['scale']=1e4
    print(config_influence)
    #influences =calc_img_wise(config_influence, model, train_dataloader, eval_dataloader)
    #config_influence['outdir']='./outdir/node_node_new/no_link'
    #calc_influence_double(config_influence, model, train_dataloader, train_dataloader)
    config_influence['outdir']='./outdir/train_train/'
    calc_all_grad_then_test(config_influence, model, train_loader, train_loader)
    config_influence['outdir']='./outdir/eval_eval/'
    calc_all_grad_then_test(config_influence, model, val_loader, val_loader)
    config_influence['outdir']='./outdir/train_eval/'
    calc_all_grad_then_test(config_influence, model, train_loader, val_loader)
    config_influence['outdir']='./outdir/eval_train/'
    calc_all_grad_then_test(config_influence, model, val_loader, train_loader)
    assert 0==1
    
    
    for epoch in range(args.epochs):
        print('Epoch [%d/%d]' % (epoch + 1, args.epochs))

        # train for one epoch
        train_loss, train_score, train_sp, train_sen = train(
            args, train_loader, model, criterion, optimizer, epoch,train_set)
        # evaluate on validation set
        val_loss, val_score, val_sp, val_sen = validate(args, val_loader, model, criterion)
        #print(val_loss, val_score, val_sp, val_sen)
        #assert 0==1
        
        if args.scheduler == 'CosineAnnealingLR':
            scheduler.step()
        elif args.scheduler == 'ReduceLROnPlateau':
            scheduler.step(val_loss)

        print('loss %.4f - score %.4f -Sp %.4f -Sen %.4f'
              % (train_loss, train_score, train_sp, train_sen))
        print('val_loss %.4f - val_score %.4f - val_Sp %.4f - val_Sen %.4f'
              % (val_loss, val_score, val_sp, val_sen))

        log['epoch'].append(epoch)
        log['loss'].append(train_loss)
        log['score'].append(train_score)
        log['val_loss'].append(val_loss)
        log['val_score'].append(val_score)

        pd.DataFrame(log).to_csv('models/%s/%s/log.csv' % (args.name,args.train_dataset), index=False)

        if val_score > best_score:
            torch.save(model.state_dict(), 'models/%s/%s/model.pth' % (args.name,args.train_dataset))
            best_loss = val_loss
            best_score = val_score
            print("=> saved best model")

    print('val_loss:  %f' % best_loss)
    print('val_score: %f' % best_score)
    
    best_losses.append(best_loss)
    best_scores.append(best_score)

    results = pd.DataFrame({
        'best_loss': best_losses + [np.mean(best_losses)],
        'best_score': best_scores + [np.mean(best_scores)],
    })

    print(results)
    results.to_csv('models/%s/%s/results.csv' % (args.name,args.train_dataset), index=False)

    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
