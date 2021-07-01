import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import MSELoss, SmoothL1Loss, _Loss
from pytorch_toolbelt.losses.functional import sigmoid_focal_loss, wing_loss
from torch.autograd import Variable

class FocalLoss(nn.Module):
      def __init__(self, focusing_param=2, balance_param=0.25):
         super(FocalLoss, self).__init__()
         self.focusing_param = focusing_param
         self.balance_param = balance_param

      def forward(self, output, target):
         cross_entropy = F.cross_entropy(output, target)
         cross_entropy_log = torch.log(cross_entropy)
         logpt = - F.cross_entropy(output, target)
         pt = torch.exp(logpt)

         focal_loss = -((1 - pt) ** self.focusing_param) * logpt
         balanced_focal_loss = self.balance_param * focal_loss

         return balanced_focal_loss


# class FocalLoss(_Loss):
#     def __init__(self, alpha=0.5, gamma=2, ignore_index=None):
#         """
#         Focal loss for multi-class problem.
#
#         :param alpha:
#         :param gamma:
#         :param ignore_index: If not None, targets with given index are ignored
#         """
#         super().__init__()
#         self.alpha = alpha
#         self.gamma = gamma
#         self.ignore_index = ignore_index
#
#     def forward(self, label_input, label_target):
#         num_classes = label_input.size(1)
#         loss = 0
#
#         # Filter anchors with -1 label from loss computation
#         if self.ignore_index is not None:
#             not_ignored = label_target != self.ignore_index
#             label_input = label_input[not_ignored]
#             label_target = label_target[not_ignored]
#
#         for cls in range(num_classes):
#             cls_label_target = (label_target == cls).long()
#             cls_label_input = label_input[:, cls]
#
#             loss += sigmoid_focal_loss(cls_label_input, cls_label_target, gamma=self.gamma, alpha=self.alpha)
#         return loss


#class FocalLoss(nn.modules.loss._WeightedLoss):
#
 #   def __init__(self, gamma=2, weight=None, size_average=None, ignore_index=-100,
 #                reduce=None, reduction='mean', balance_param=0.25):
 #       super(FocalLoss, self).__init__(weight, size_average, reduce, reduction)
 #       self.gamma = gamma
 #       self.weight = weight
 #       self.size_average = size_average
 #       self.ignore_index = ignore_index
 #       self.balance_param = balance_param

  #  def forward(self, input, target):
        # inputs and targets are assumed to be BatchxClasses
        #assert len(input.shape) == len(target.shape)
        #assert input.size(0) == target.size(0)
        #assert input.size(1) == target.size(1)

  #      weight = Variable(self.weight)

        # compute the negative likelyhood
   #     logpt = - F.binary_cross_entropy_with_logits(input, target, pos_weight=weight, reduction=self.reduction)
   #     pt = torch.exp(logpt)

        # compute the loss
   #     focal_loss = -((1 - pt) ** self.gamma) * logpt
    #    balanced_focal_loss = self.balance_param * focal_loss
    #    return balanced_focal_loss


class FocalLoss4(nn.Module):
     def __init__(self, gamma=2, alpha=1, size_average=True):
         super(FocalLoss, self).__init__()
         self.gamma = gamma
         self.alpha = alpha
         self.size_average = size_average
         self.elipson = 0.000001
 
     def forward(self, logits, labels):
         """
#         cal culates loss
#         logits: batch_size * labels_length * seq_length
#         labels: batch_size * seq_length
#         """
         if labels.dim() > 2:
             labels = labels.contiguous().view(labels.size(0), labels.size(1), -1)
             labels = labels.transpose(1, 2)
             labels = labels.contiguous().view(-1, labels.size(2)).squeeze()
         if logits.dim() > 3:
             logits = logits.contiguous().view(logits.size(0), logits.size(1), logits.size(2), -1)
             logits = logits.transpose(2, 3)
             logits = logits.contiguous().view(-1, logits.size(1), logits.size(3)).squeeze()
         assert (logits.size(0) == labels.size(0))
         assert (logits.size(2) == labels.size(1))
         batch_size = logits.size(0)
         labels_length = logits.size(1)
         seq_length = logits.size(2)

         # transpose labels into labels onehot
         new_label = labels.unsqueeze(1)
         label_onehot = torch.zeros([batch_size, labels_length, seq_length]).scatter_(1, new_label, 1)
         # label_onehot = label_onehot.permute(0, 2, 1) # transpose, batch_size * seq_length * labels_length

         # calculate log
         log_p = F.log_softmax(logits)
         pt = label_onehot * log_p
         sub_pt = 1 - pt
         fl = -self.alpha * (sub_pt) ** self.gamma * log_p
         if self.size_average:
             return fl.mean()
         else:
             return fl.sum()
