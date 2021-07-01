from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import pretrainedmodels
from efficientnet_pytorch import EfficientNet


def get_model(model_name='efficientnet', num_outputs=5, pretrained=True,
              freeze_bn=False, dropout_p=0, **kwargs):

    if 'efficientnet' in model_name:
        model = EfficientNet.from_pretrained('efficientnet-b2')
        feature = model._fc.in_features
        model._fc = nn.Linear(in_features=feature, out_features=5, bias=True)

    if freeze_bn:
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.weight.requires_grad = False
                m.bias.requires_grad = False

    return model
