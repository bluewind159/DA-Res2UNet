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

from lib.dataset_fusion import Dataset
from lib.models.model_factory import get_model
from lib.utils import *
from lib.metrics import *
from lib.losses import *
from lib.optimizers import *
from lib.preprocess import preprocess
from lib.models.res2unet import Res2UNet,Res2UNet_add
from lib.models.nestedUNet import NestedUNet
import cv2
from misc_functions import *
import random
from pengzhang import Morphology_Erode,Morphology_Dilate
import warnings
warnings.filterwarnings('ignore')
def auc_calculate(labels,preds,n_bins=100):
    postive_len = sum(labels)
    negative_len = len(labels) - postive_len
    total_case = postive_len * negative_len
    pos_histogram = [0 for _ in range(n_bins)]
    neg_histogram = [0 for _ in range(n_bins)]
    bin_width = 1.0 / n_bins
    for i in range(len(labels)):
        
        nth_bin = int(preds[i]/bin_width)
        if labels[i]==1:
            pos_histogram[nth_bin] += 1
        else:
            neg_histogram[nth_bin] += 1
    accumulated_neg = 0
    satisfied_pair = 0
    for i in range(n_bins):
        satisfied_pair += (pos_histogram[i]*accumulated_neg + pos_histogram[i]*neg_histogram[i]*0.5)
        accumulated_neg += neg_histogram[i]

    return satisfied_pair / float(total_case)

def dice_coeff(pred, target):
    smooth = 1.
    num = pred.size(0)
    m1 = pred.view(num, -1)  # Flatten
    m2 = target.view(num, -1)  # Flatten
    intersection = (m1 * m2).sum()
 
    return (2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth)

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()
        self.epsilon = 1e-5
    def forward(self, predict, target):
        assert predict.size() == target.size(), "the size of predict and target must be equal."
        num = predict.size(0)
        pre = torch.sigmoid(predict).view(num, -1)
        tar = target.view(num, -1)
        intersection = (pre * tar).sum(-1).sum()
        union = (pre + tar).sum(-1).sum()
        score = 1 - 2 * (intersection + self.epsilon) / (union + self.epsilon)
        return score
        
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
    parser.add_argument('--epochs', default=100, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch_size', default=1, type=int,
                        metavar='N', help='mini-batch size (default: 32)')
    parser.add_argument('--img_size', default=512, type=int,
                        help='input image size (default: 288)')
    parser.add_argument('--optimizer', default='Adam')
    parser.add_argument('--pred_type', default='classification',
                        choices=['classification', 'regression', 'multitask'])
    parser.add_argument('--scheduler', default='CosineAnnealingLR',
                        choices=['CosineAnnealingLR', 'ReduceLROnPlateau','None'])
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
    for i, (input,input_new, target,target_d) in tqdm(enumerate(train_loader), total=len(train_loader)):
        #input,mu,sigma=input
        input = input.cuda()
        #input_new,mu_new,sigma_new=input_new
        input_new=input_new.cuda()
        index=random.randint(0,len(train_loader)-1)
        afafaf=random.random()
        if afafaf>0.8:
            add_noise_mark=True
        else:
            add_noise_mark=False
        im_add,_,label_add,label_add_d=train_set[index]
        #im_add,mu_add,sigma_add=im_add
        add_mark1=label_add
        m_add=(label_add>0).long()
        anti_m_add=(~(label_add>0)).long()
        add_mark2=im_add*m_add
        #with torch.no_grad():
        #    add_mark=model(add_image.unsqueeze(0).cuda())
        #    add_mark=add_mark[0]
        block_size=16
        desired_size=input.size(-1)
        im_acc1=torch.zeros(add_mark1.size())
        im_acc2=torch.zeros(add_mark2.size())
        assert desired_size%block_size==0
        for i in range(int(desired_size/block_size)):
            for j in range(int(desired_size/block_size)):
                acc=random.random()
                bblock = add_mark1[:,i*block_size:(i+1)*block_size,j*block_size:(j+1)*block_size]
                im_block=add_mark2[:,i*block_size:(i+1)*block_size,j*block_size:(j+1)*block_size]
                if acc>0.3:
                    im_acc1[:,i*block_size:(i+1)*block_size,j*block_size:(j+1)*block_size]=bblock*0
                    im_acc2[:,i*block_size:(i+1)*block_size,j*block_size:(j+1)*block_size]=im_block*0
                else:
                    aaaa=random.random()
                    trans=random.random()
                    if trans>0.5:
                        bblock=bblock.transpose(2,1)
                        im_block=im_block.transpose(2,1)
                    r=random.randint(0,15)
                    c=random.randint(0,15)
                    bblock=torch.roll(bblock,r,dims=1)
                    bblock=torch.roll(bblock,c,dims=2)
                    im_block=torch.roll(im_block,r,dims=1)
                    im_block=torch.roll(im_block,c,dims=2)
                    im_acc1[:,i*block_size:(i+1)*block_size,j*block_size:(j+1)*block_size]=bblock
                    im_acc2[:,i*block_size:(i+1)*block_size,j*block_size:(j+1)*block_size]=im_block
        if not torch.equal(label_add,target):
            if add_noise_mark:
                im_acc1=im_acc1.unsqueeze(0).cuda()
                im_acc2=im_acc2.unsqueeze(0).cuda()
                target = target.cuda()
                #mask=(target>0).long().cuda()
                #anti_mask=(~(target>0)).long().cuda()
                mask=(im_acc1>0).long().cuda()
                anti_mask=(~(im_acc1>0)).long().cuda()
                #im_acc=im_acc*anti_mask
                wwwq=random.random()
                if wwwq>=0.5:
                    input_new=input_new*anti_mask+im_acc2.cuda()
                else:
                    input_new=input_new+im_acc1
        target = target.cuda()
        target_d = target_d.cuda()
        output = model(input_new)
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
        if False:
            image = qwe1[0].cpu()
            image = image.data.cpu().numpy()
            image = np.uint8(image*sigma.numpy()+mu.numpy()).transpose(1,2,0)
            image = cv2.resize(image, (1024, 1024))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image2 = qwe2[0].cpu()
            image2 = image2.data.cpu().numpy()
            image2 = np.uint8(image2*sigma.numpy()+mu.numpy()).transpose(1,2,0)
            image2 = cv2.resize(image2, (1024, 1024))
            image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
            image3 = im_acc1[0]
            image3 = image3.data.cpu().numpy()
            image3 = np.uint8(image3*sigma.numpy()+mu.numpy()).transpose(1,2,0)
            image3 = cv2.resize(image3, (1024, 1024))
            image3 = cv2.cvtColor(image3, cv2.COLOR_BGR2RGB)
            image4 = im_acc2[0]
            image4 = image4.data.cpu().numpy()
            image4 = np.uint8(image4*sigma.numpy()+mu.numpy()).transpose(1,2,0)
            image4 = cv2.resize(image4, (1024, 1024))
            image4 = cv2.cvtColor(image4, cv2.COLOR_BGR2RGB)
            plt.subplot(1,4,1)
            plt.imshow(image3)
            plt.title('noise1')
            plt.subplot(1,4,2)
            plt.imshow(image2)
            plt.title('image add noise1')
            plt.subplot(1,4,3)
            plt.imshow(image4)
            plt.title('noise2')
            plt.subplot(1,4,4)
            plt.imshow(image)
            plt.title('image add noise2')
            plt.show()
            assert 0==1
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
    #AUC = AverageMeter()
    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for i, (input,input_new, target,target_d) in tqdm(enumerate(val_loader), total=len(val_loader)):
            input = input.cuda()
            target = target.cuda()
            output = model(input)
            loss = criterion(output, target)
            #A=auc_calculate(target.view(-1).data.cpu().numpy(),output.view(-1).data.cpu().numpy())
            #AUC.update(A,input.size(0))
            output233=(output >0.5).long()
            score,Sp,Sen,out_RGB = accuracy(output233, target)
            losses.update(loss.item(), input.size(0))
            scores.update(score, input.size(0))
            specificity.update(Sp.item(), input.size(0))
            sensitivity.update(Sen, input.size(0))
            if False:
                image = output[0].cpu()
                image = image.data.cpu().numpy()
                image = np.uint8(image*255).transpose(1,2,0)
                image = cv2.resize(image, (1024, 1024))
                #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                #cv2.imshow('image',image)
                #cv2.waitKey(0)
                #assert 0==1
                cv2.imwrite('results/'+args.train_dataset+'/'+str(i)+'.png',image)
                cv2.imwrite('results/'+args.train_dataset+'/'+str(i)+'RGB.png',out_RGB)
    return losses.avg, scores.avg, specificity.avg, sensitivity.avg, 0#AUC.avg


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
    ])
    train_label_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
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
        transform=[train_transform,train_label_transform],
        fusion=True)
    #print('not fusion the data')
        
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
        
    test_set = Dataset(
        test_img_paths,
        test_label_paths,
        transform=[val_transform,val_label_transform])
        
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4)

    model = Res2UNet_add(1,1)
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
    
    #test_loss, test_score, test_sp, test_sen = validate(args, test_loader, model, criterion)
    #assert 0==1
    for epoch in range(args.epochs):
        print('Epoch [%d/%d]' % (epoch + 1, args.epochs))

        # train for one epoch
        train_loss, train_score, train_sp, train_sen = train(
            args, train_loader, model, criterion, optimizer, epoch,train_set)
        # evaluate on validation set
        #val_loss, val_score, val_sp, val_sen, _ = validate(args, val_loader, model, criterion)
        # 
        test_loss, test_score, test_sp, test_sen, test_AUC = validate(args, test_loader, model, criterion)
        #print('test_loss %.4f - test_score %.4f - test_Sp %.4f - test_Sen %.4f - test_AUC %.4f'
        #      % (test_loss, test_score, test_sp, test_sen, test_AUC))
        #assert 0==1
        
        if args.scheduler == 'CosineAnnealingLR':
            scheduler.step()
        elif args.scheduler == 'ReduceLROnPlateau':
            scheduler.step(val_loss)

        print('loss %.4f - score %.4f -Sp %.4f -Sen %.4f'
              % (train_loss, train_score, train_sp, train_sen))
        #print('val_loss %.4f - val_score %.4f - val_Sp %.4f - val_Sen %.4f'
        #      % (val_loss, val_score, val_sp, val_sen))
        print('test_loss %.4f - test_score %.4f - test_Sp %.4f - test_Sen %.4f - test_AUC %.4f'
              % (test_loss, test_score, test_sp, test_sen, test_AUC))

        log['epoch'].append(epoch)
        log['loss'].append(train_loss)
        log['score'].append(train_score)
        log['val_loss'].append(test_loss)
        log['val_score'].append(test_score)

        pd.DataFrame(log).to_csv('models/%s/%s/log.csv' % (args.name,args.train_dataset), index=False)

        if test_score > best_score:
            torch.save(model.state_dict(), 'models/%s/%s/model.pth' % (args.name,args.train_dataset))
            best_loss = test_loss
            best_score = test_score
            best_score_test = test_score
            best_sp=test_sp
            best_sen=test_sen
            print("=> saved best model")
        print('best_loss %.4f - best_score %.4f - best_Sp %.4f - best_Sen %.4f'
              % (best_loss, best_score_test, best_sp, best_sen))

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
