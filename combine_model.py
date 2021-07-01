from torch import nn, Tensor
import torch
import cv2
import numpy as np
import torchvision
from misc_functions import *
import matplotlib.pyplot as plt
from torchvision import datasets, models, transforms
from PIL import Image

class combine_model(nn.Module):
    def __init__(self, generator,exact_model):
        super(combine_model, self).__init__()
        self.generator=generator.eval()
        self.exact_model=exact_model.eval()
        self.rank=0
        self.transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                    ])
        self.unnormalize = NormalizeInverse([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    def forward(self, batch_size, latent_dim) -> Tensor:
        #filename='generate_data/resized_train/16_left.jpeg'
        #filename='CHASE/test/image/Image_12L.jpg'
        #im=cv2.imread(filename)
        #im = cv2.resize(im, (512, 512))
        #im = Image.fromarray(im)
        #im=self.transform(im)
        #generated_images=im.unsqueeze(0).cuda()
        print('model_ext_test')
        G=self.generator
        latents = torch.randn(batch_size, latent_dim).cuda(self.rank)
        generated_images = G(latents)
        generated_images = generated_images.clamp_(0., 1.)
        #generated_images = generated_images
        #generated_images = self.transform(generated_images)
        output = self.exact_model(generated_images)
        #output = (output>0.5).long()
        label = output[0].data.cpu().numpy()
        label = np.uint8(label * 255).transpose(1,2,0)
        label = cv2.resize(label, (1024, 1024))
        #image233 = self.unnormalize(generated_images[0].cpu())
        image233 = generated_images[0].cpu()
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
        plt.show()
        assert 0==1