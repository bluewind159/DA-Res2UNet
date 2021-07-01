import torch
from sklearn import metrics
import numpy as np
from sklearn.utils.multiclass import type_of_target
import cv2
# def quadratic_weighted_kappa(y_pred, y_true):
#     if torch.is_tensor(y_pred):
#         y_pred = y_pred.data.cpu().numpy()
#     if torch.is_tensor(y_true):
#         y_true = y_true.data.cpu().numpy()
#     if y_pred.shape[1] == 1:
#         y_pred = y_pred[:, 0]
#     else:
#         y_pred = np.argmax(y_pred, axis=1)
#     return metrics.cohen_kappa_score(y_pred, y_true, weights='quadratic')


# def accuracy(y_pred, y_actual, topk=(1,)):
#     """Computes the precision@k for the specified values of k"""
#     maxk = max(topk)
#     batch_size = y_actual.size(0)
#
#     _, pred = y_pred.topk(maxk, 1, True, True)
#     pred = pred.t()
#     correct = pred.eq(y_actual.view(1, -1).expand_as(pred))
#
#     res = []
#     for k in topk:
#         correct_k = correct[:k].view(-1).float().sum(0)
#         res.append(correct_k.mul_(100.0 / batch_size))
#
#     return res
#
#
# def accuracy(output, target, topk=(1,)):
#     """Computes the accuracy over the k top predictions for the specified values of k"""
#     with torch.no_grad():
#         maxk = max(topk)
#         batch_size = target.size(0)
#
#         _, pred = output.topk(maxk, 1, True, True)
#         pred = pred.t()
#         correct = pred.eq(target.view(1, -1).expand_as(pred))
#
#         res = []
#         for k in topk:
#             correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
#             res.append(correct_k.mul_(100.0 / batch_size))
#         return res

def accuracy(y_pred, y_true):
    if torch.is_tensor(y_pred):
        y_pred = y_pred.data.cpu().numpy()
    if torch.is_tensor(y_true):
        y_true = y_true.data.cpu().numpy()
    # True Positive:��y_true��y_pred��ͬʱΪ1�ĸ���
    TP = np.sum(np.logical_and(np.equal(y_true, 1), np.equal(y_pred, 1)))   # 10
    #TP = np.sum(np.multiply(y_true, y_pred)) #ͬ������ʵ�ּ���TP
    # False Positive:��y_true��Ϊ0������y_pred�б�ʶ��Ϊ1�ĸ���
    FP = np.sum(np.logical_and(np.equal(y_true, 0), np.equal(y_pred, 1)))   #  0
    # False Negative:��y_true��Ϊ1������y_pred�б�ʶ��Ϊ0�ĸ���
    FN = np.sum(np.logical_and(np.equal(y_true, 1), np.equal(y_pred, 0)))  # 6
    # True Negative:��y_true��y_pred��ͬʱΪ0�ĸ���
    TN = np.sum(np.logical_and(np.equal(y_true, 0), np.equal(y_pred, 0)))  # 34
    R=np.logical_and(np.equal(y_true, 1), np.equal(y_pred, 0))
    G=np.logical_and(np.equal(y_true, 1), np.equal(y_pred, 1))
    B=np.logical_and(np.equal(y_true, 0), np.equal(y_pred, 1))
    R=np.uint8(R[0]*255)
    G=np.uint8(G[0]*255)
    B=np.uint8(B[0]*255)
    image=np.concatenate((R,G,B),axis=0).transpose(1,2,0)
    # ��������õ���ֵ����A��P��R��F1
    #print('TP:',TP,'FP:',FP,'FN:',FN,'TN:',TN)
    A=(TP+TN)/(TP+FP+FN+TN) #y_pred��y_ture��ͬʱΪ1��0
    Sp=TN/(TN+FP) #y_pred��Ϊ1��Ԫ��ͬʱ��y_true��ҲΪ1
    Sen=TP/(TP+FN) #y_true��Ϊ1��Ԫ��ͬʱ��y_pred��ҲΪ1
    return A,Sp,Sen,image
