#!/usr/bin/env python
# coding: utf-8
#
# Author:   Kazuto Nakashima
# URL:      http://kazuto1011.github.io
# Created:  2017-05-18

from __future__ import print_function

import copy

import click
import cv2
import numpy as np
import torch
from torch.autograd import Variable
from torchvision import models, transforms

from grad_cam import (BackPropagation, Deconvolution, GradCAM,
                      GuidedBackPropagation)
import models.se_resnet
model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))


def to_var(image):
    return Variable(image.unsqueeze(0), volatile=False, requires_grad=True)


def save_gradcam(filename, gcam, raw_image):
    h, w, _ = raw_image.shape
    gcam = cv2.resize(gcam, (w, h))
    gcam = cv2.applyColorMap(np.uint8(gcam * 255.0), cv2.COLORMAP_JET)
    gcam = gcam.astype(np.float) + raw_image.astype(np.float)
    gcam = gcam / gcam.max() * 255.0
    cv2.imwrite(filename, np.uint8(gcam))


@click.command()
@click.option('--image-path', type=str, required=True)
@click.option('--arch', type=str, required=True)
@click.option('--topk', type=int, default=3)
@click.option('--cuda/--no-cuda', default=True)
@click.option('--target-layer', type=str, required=True)
def main(image_path, arch, topk, cuda, target_layer):

    CONFIG = {
        'resnet152': {
            'target_layer': 'layer4.2',
            'input_size': 224
        },
        'vgg19': {
            'target_layer': 'features.36',
            'input_size': 224
        },
        'vgg19_bn': {
            'target_layer': 'features.52',
            'input_size': 224
        },
        'inception_v3': {
            'target_layer': 'Mixed_7c',
            'input_size': 299
        },
        'densenet201': {
            'target_layer': 'features.denseblock4',
            'input_size': 224
        },
        # Add your model
        'se_resnet': {
            'target_layer': target_layer,
            'input_size': 32
        },
    }.get(arch)

    cuda = cuda and torch.cuda.is_available()

    if cuda:
        current_device = torch.cuda.current_device()
        print('Running on the GPU:', torch.cuda.get_device_name(current_device))
    else:
        print('Running on the CPU')

    # Synset words
    classes = list()
    with open('samples/synset_words.txt') as lines:
        for line in lines:
            line = line.strip().split(' ', 1)[1]
            line = line.split(', ', 1)[0].replace(' ', '_')
            classes.append(line)

    # Model
    #model = models.__dict__[arch](pretrained=True)
    model = models.se_resnet.se_resnet50(num_classes=100)

    # Image
    raw_image = cv2.imread(image_path)[..., ::-1]
    raw_image = cv2.resize(raw_image, (CONFIG['input_size'], ) * 2)
    image = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])(raw_image)

    if cuda:
        model.cuda()
        image = image.cuda()

    # =========================================================================
    print('Grad-CAM')
    # =========================================================================
    gcam = GradCAM(model=model)
    probs, idx = gcam.forward(to_var(image))

    for i in range(0, topk):
        gcam.backward(idx=idx[i])
        output = gcam.generate(target_layer=CONFIG['target_layer'])

        #save_gradcam('results/{}_gcam_{}.png'.format(classes[idx[i]], arch), output, raw_image)  # NOQA
        save_gradcam('results/{}.png'.format(target_layer), output, raw_image)  # NOQA
        print('[{:.5f}] {}'.format(probs[i], classes[idx[i]]))


if __name__ == '__main__':
    main()
