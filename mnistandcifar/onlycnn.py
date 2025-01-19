import torch
import torch.nn as nn
import torch
import torch.nn
import torch.optim as optim
import torchvision.transforms as Transform
from torchvision.utils import save_image


import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import numpy as np
import os
from argparse import ArgumentParser
from tqdm import tqdm



from utils2 import save_model, denorm, plot_loss, plot_classifier_loss, show_process
from utils2 import generation_by_attributes, get_random_label

class Discriminator(nn.Module):
    """ ACGAN discriminator.
    
    A modified version of the DCGAN discriminator. Aside from a discriminator
    output, DCGAN discriminator also classifies the class of the input image 
    using a fully-connected layer.

    Attributes:
    	num_classes: number of classes the discriminator needs to classify.
    	conv_layers: all convolutional layers before the last DCGAN layer. 
    				 This can be viewed as an feature extractor.
    	discriminator_layer: last layer of DCGAN. Outputs a single scalar.
    	bottleneck: Layer before classifier_layer.
    	classifier_layer: fully conneceted layer for multilabel classifiction.
			
    """
    def __init__(self, num_classes):
        """ Initialize Discriminator Class with num_classes."""
        super(Discriminator, self).__init__()

        self.num_classes = num_classes
        self.conv_layers = nn.Sequential(
                    nn.Conv2d(in_channels = 1, 
                             out_channels = 128, 
                             kernel_size = 2,
                             stride = 2,
                             padding = 1,
                             bias = False),
                    nn.LeakyReLU(0.2, inplace = True),
                    nn.Conv2d(in_channels = 128, 
                             out_channels = 256, 
                             kernel_size = 3,
                             stride = 2,
                             padding = 1,
                             bias = False),
                    nn.BatchNorm2d(256),
                    nn.LeakyReLU(0.2, inplace = True),
                    nn.Conv2d(in_channels = 256, 
                             out_channels = 512, 
                             kernel_size = 4,
                             stride = 2,
                             padding = 1,
                             bias = False),
                    nn.BatchNorm2d(512),
                    nn.LeakyReLU(0.2, inplace = True),
                   
                    )   
      
        self.bottleneck = nn.Sequential(
                    nn.Conv2d(in_channels = 512, 
                             out_channels = 256, 
                             kernel_size = 4,
                             stride = 1),
                    nn.BatchNorm2d(256),
                    nn.LeakyReLU(0.2)
                    )
        self.classifier_layer = nn.Sequential(
                    nn.Linear(256, self.num_classes),
                    nn.Softmax()
                    )

        return
    
    def forward(self, _input):
        """ Defines a forward pass of a discriminator.
        Args:
            _input: A batch of image tensors. Shape: N * 3 * 64 *64
        
        Returns:
            discrim_output: Value between 0-1 indicating real or fake. Shape: N * 1
            aux_output: Class scores for each class. Shape: N * num_classes
        """

        features = self.conv_layers(_input)  
        flatten = self.bottleneck(features).squeeze()
        aux_output = self.classifier_layer(flatten) # Outputs probability for each class label
        return  aux_output





device='cuda'

transform = transforms.Compose([Transform.ToTensor(),
                                Transform.Normalize((0.5,), (0.5,))])

dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
batch_size=128
# Define a DataLoader for shuffling and batching
shuffler = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
num_classes=10
D = Discriminator(num_classes = num_classes).to(device)

D_optim = optim.Adam(D.parameters(), betas = [ 0.5, 0.999], lr = 0.0002)


criterion1 = torch.nn.CrossEntropyLoss()
d_log, g_log, classifier_log = [], [], []
for step_i in tqdm(range(1, 50000 + 1), desc="Training Progress", unit="step"):
    
    real_label = torch.ones(batch_size).to(device)
    
    
    # Train discriminator
    real_img, hair_tags = next(iter(shuffler))  
    real_img, hair_tags = real_img.to(device), hair_tags.to(device)
    num_classes = 10  # Number of MNIST classes
    real_tag = torch.nn.functional.one_hot(hair_tags, num_classes=num_classes).float()  # Shape: [batch_size, num_classes]


            
    real_predict = D(real_img)
  
   
    real_classifier_loss = criterion1(real_predict, real_tag)
    
        
    D_loss = real_classifier_loss
    D_optim.zero_grad()
    D_loss.backward()
    D_optim.step()
    checkpoint_dir="/samples/cnn"
    d_log.append(D_loss.item())
        
    if step_i % 1000 == 0:
      
        save_model(model = D, optimizer = D_optim, step = step_i, log = tuple(d_log), 
                    file_path = os.path.join(checkpoint_dir, 'D_{}.ckpt'.format(step_i)))

    