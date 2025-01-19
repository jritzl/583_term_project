from torchvision import datasets, transforms
from torch.utils.data import DataLoader
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


from model3 import Generator, Discriminator
from utils3 import save_model, denorm, plot_loss, plot_classifier_loss, show_process
from utils3 import generation_by_attributes, get_random_label


num_classes = 10  # MNIST has 10 classes
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load test dataset

# Define a DataLoader for the test dataset
G = Generator(latent_dim = 100, class_dim = 2).to(device)
classifier_weights_path = '/checkpoints/models1/G_40000.ckpt'  # Replace with your path
G.load_state_dict(torch.load(classifier_weights_path),strict=False)

prev_state = torch.load(classifier_weights_path)

G = Generator(100, 2).to(device)
G.load_state_dict(prev_state['model'])
G.eval()



hair_tag = torch.zeros(16, 2).to(device)


for j in range(2):
    start_idx = j * 8
    hair_tag[start_idx:start_idx+8, j] = 1


z = torch.randn(16, 100).to(device)

output = G(z, hair_tag)

file_path = 'hair.png'
save_image(denorm(output), os.path.join("/content/", file_path))
