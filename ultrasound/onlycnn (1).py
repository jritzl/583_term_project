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



from utils3 import save_model, denorm, plot_loss, plot_classifier_loss, show_process
from utils3 import generation_by_attributes, get_random_label

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
                             kernel_size = 4,
                             stride = 2,
                             padding = 1,
                             bias = False),
                    nn.LeakyReLU(0.2, inplace = True),
                    nn.Conv2d(in_channels = 128, 
                             out_channels = 256, 
                             kernel_size = 4,
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

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

import zipfile
import os
from PIL import Image
import numpy as np

class UltrasoundDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        """
        Args:
            images (numpy array): Array of image data (N, H, W, C).
            labels (numpy array): Array of labels (N,).
            transform (callable, optional): Optional transform to be applied
                                            on a sample.
        """
        self.images = images
        self.labels = labels
        self.transform = transform
        self.label_to_idx = {'benign': 0, 'malignant': 1}  # Map labels to integers

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Get image and label
        image = self.images[idx]
        label = self.label_to_idx[self.labels[idx]]

        # Apply any transforms if provided
        if self.transform:
            image = self.transform(image)

        # Convert to PyTorch tensors
        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)  # (C, H, W)
        label = torch.tensor(label, dtype=torch.long)

        return image, label


# Path to the ZIP file
zip_path = "/content/us-dataset.zip"

batch_size=32

# Initialize lists to store images and labels
images = []
labels = []

# Define a target image size
target_size = (32, 32)  # Resize all images to 224x224

# Open the ZIP file
with zipfile.ZipFile(zip_path, 'r') as z:
    # List all files in the ZIP
    file_list = z.namelist()
    
    # Filter BMP files inside 'originals/benign' and 'originals/malignant'
    bmp_files = [f for f in file_list if f.startswith("originals/") and f.endswith(".bmp")]
    
    for file_path in bmp_files:
        # Extract the label from the folder name (benign or malignant)
        label = os.path.basename(os.path.dirname(file_path))
        
        # Open the BMP file directly from the ZIP
        with z.open(file_path) as bmp_file:
            img = Image.open(bmp_file)
            
            # Resize the image to the target size and normalize pixel values
            img = img.resize(target_size)
            img_array = np.array(img) / 255.0  # Normalize to [0, 1]
            
            # Ensure all images have 3 channels (convert grayscale to RGB if needed)
            if img_array.ndim == 2:  # Grayscale
                img_array = np.stack([img_array], axis=-1)
            
            images.append(img_array)
            labels.append(label)
# Convert lists to NumPy arrays
images = np.array(images)  # Shape: (N, 224, 224, 3)
labels = np.array(labels)  # Shape: (N,)
images = np.concatenate([images[:75], images[125:250]], axis=0)
labels = np.concatenate([labels[:75], labels[125:250]], axis=0)
images = np.array(images)  # Shape: (N, 224, 224, 3)
labels = np.array(labels)  # Shape: (N,)

print(f"Loaded {len(images)} images with shape: {images[0].shape}")
print(f"Labels: {np.unique(labels)}")






# Instantiate the dataset
dataset = UltrasoundDataset(images, labels)

# Create a DataLoader
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Iterate through the DataLoader
for batch_idx, (images_batch, labels_batch) in enumerate(dataloader):
    print(f"Batch {batch_idx}:")
    print(f"Images shape: {images_batch.shape}")  # (batch_size, 3, H, W)
    print(f"Labels shape: {labels_batch.shape}")  # (batch_size,)
    break



shuffler=DataLoader(dataset, batch_size=32, shuffle=True)
device='cuda'

# transform = transforms.Compose([Transform.ToTensor(),
#                                 Transform.Normalize((0.5,), (0.5,))])

# dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
# batch_size=128
# # Define a DataLoader for shuffling and batching
# shuffler = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
num_classes=2
D = Discriminator(num_classes = num_classes).to(device)

D_optim = optim.Adam(D.parameters(), betas = [ 0.5, 0.999], lr = 0.0002)


criterion1 = torch.nn.CrossEntropyLoss()
d_log, g_log, classifier_log = [], [], []
for step_i in tqdm(range(1, 50000 + 1), desc="Training Progress", unit="step"):
    
    real_label = torch.ones(batch_size).to(device)
    
    
    # Train discriminator
    real_img, hair_tags = next(iter(shuffler))  
    real_img, hair_tags = real_img.to(device), hair_tags.to(device)
    num_classes = 2 # Number of MNIST classes
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

    