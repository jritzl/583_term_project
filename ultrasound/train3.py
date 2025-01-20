import torch
import torch.nn
import torch.optim as optim
import torchvision.transforms as Transform
from torchvision.utils import save_image


import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

import numpy as np
import os
from argparse import ArgumentParser
from tqdm import tqdm


from model3 import Generator, Discriminator
from utils3 import save_model, denorm, plot_loss, plot_classifier_loss, show_process
from utils3 import generation_by_attributes, get_random_label

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










parser = ArgumentParser()
parser.add_argument('-d', '--device', help = 'Device to train the model on', 
                    default = 'cuda', choices = ['cuda', 'cpu'], type = str)
parser.add_argument('-i', '--iterations', help = 'Number of iterations to train ACGAN', 
                    default = 40005, type = int)
parser.add_argument('-b', '--batch_size', help = 'Training batch size',
                    default = 128, type = int)
parser.add_argument('-t', '--train_dir', help = 'Training data directory', 
                    default = '../data', type = str)
parser.add_argument('-s', '--sample_dir', help = 'Directory to store generated images', 
                    default = '../samples', type = str)
parser.add_argument('-c', '--checkpoint_dir', help = 'Directory to save model checkpoints', 
                    default = '../checkpoints', type = str)
parser.add_argument('--sample', help = 'Sample every _ steps', 
                    default = 1000, type = int)
parser.add_argument('--check', help = 'Save model every _ steps', 
                    default = 2000, type = int)
parser.add_argument('--lr', help = 'Learning rate of ACGAN. Default: 0.0002', 
                    default = 0.0002, type = float)
parser.add_argument('--beta', help = 'Momentum term in Adam optimizer. Default: 0.5', 
                    default = 0.5, type = float)
parser.add_argument('--aux', '--classification_weight', help = 'Classification loss weight. Default: 1',
                    default = 1, type = float)
args = parser.parse_args()

if args.device == 'cuda' and not torch.cuda.is_available():
    print("Your device currenly doesn't support CUDA.")
    exit()
print('Using device: {}'.format(args.device))

def main():
    batch_size = args.batch_size
    iterations =  args.iterations
    device = args.device
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
    
    num_classes = 10
    latent_dim = 100
    
    config = 'ACGAN-batch_size-[{}]-steps-[{}]'.format(batch_size, iterations)
    print('Configuration: {}'.format(config))
    
    
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


    random_sample_dir = '../{}/{}/random_generation'.format(args.sample_dir, config)
    fixed_attribute_dir = '../{}/{}/fixed_attributes'.format(args.sample_dir, config)
    checkpoint_dir = '../{}/{}'.format(args.checkpoint_dir, config)
    
    if not os.path.exists(random_sample_dir):
    	os.makedirs(random_sample_dir)
    if not os.path.exists(fixed_attribute_dir):
    	os.makedirs(fixed_attribute_dir)
    if not os.path.exists(checkpoint_dir):
    	os.makedirs(checkpoint_dir)
        


   


    
    G = Generator(latent_dim = latent_dim, class_dim = num_classes).to(device)
    D = Discriminator(num_classes = num_classes).to(device)

    G_optim = optim.Adam(G.parameters(), betas = [args.beta, 0.999], lr = args.lr)
    D_optim = optim.Adam(D.parameters(), betas = [args.beta, 0.999], lr = args.lr)
    
    d_log, g_log, classifier_log = [], [], []
    criterion1 = torch.nn.CrossEntropyLoss()
    criterion2 = torch.nn.BCELoss()
    for step_i in tqdm(range(1, iterations + 1), desc="Training Progress", unit="step"):
        
        real_label = torch.ones(batch_size).to(device)
        fake_label = torch.zeros(batch_size).to(device)
        
        # Train discriminator
        real_img, hair_tags = next(iter(shuffler))  
        real_img, hair_tags = real_img.to(device), hair_tags.to(device)
        num_classes = 10  # Number of MNIST classes
        real_tag = torch.nn.functional.one_hot(hair_tags, num_classes=num_classes).float()  # Shape: [batch_size, num_classes]

        z = torch.randn(batch_size, latent_dim).to(device)
        fake_tag = get_random_label(batch_size = batch_size, 
                                    hair_classes = num_classes, hair_prior = None,
                                    eye_classes = None, eye_prior = None).to(device)
        fake_img = G(z, fake_tag).to(device)
                
        real_score, real_predict = D(real_img)
        fake_score, fake_predict = D(fake_img)

        
        
        real_discrim_loss = criterion2(real_score, real_label)
        fake_discrim_loss = criterion2(fake_score, fake_label)

        real_classifier_loss = criterion1(real_predict, real_tag)
        
        discrim_loss = (real_discrim_loss + fake_discrim_loss) * 0.5
        classifier_loss = real_classifier_loss * 0.5
        
        classifier_log.append(classifier_loss.item())
            
        D_loss = discrim_loss + classifier_loss
        D_optim.zero_grad()
        D_loss.backward()
        D_optim.step()

        # Train generator
        z = torch.randn(batch_size, latent_dim).to(device)
        fake_tag = get_random_label(batch_size = batch_size, 
                                    hair_classes = num_classes, hair_prior = None,
                                    eye_classes = None, eye_prior = None).to(device)
        fake_img = G(z, fake_tag).to(device)
        
        fake_score, fake_predict = D(fake_img)
        
        discrim_loss = criterion2(fake_score, real_label)
        classifier_loss = criterion1(fake_predict, fake_tag)
        
        G_loss = classifier_loss + discrim_loss
        G_optim.zero_grad()
        G_loss.backward()
        G_optim.step()
            
        ########## Updating logs ##########
        d_log.append(D_loss.item())
        g_log.append(G_loss.item())
        #show_process(total_steps = iterations, step_i = step_i,
        #			 g_log = g_log, d_log = d_log, classifier_log = classifier_log)

        ########## Checkpointing ##########

        if step_i == 1:
            save_image(denorm(real_img[:64,:,:,:]), os.path.join(random_sample_dir, 'real.png'))
        if step_i % args.sample == 0:
            save_image(denorm(fake_img[:64,:,:,:]), os.path.join(random_sample_dir, 'fake_step_{}.png'.format(step_i)))
            
        if step_i % args.check == 0:
            save_model(model = G, optimizer = G_optim, step = step_i, log = tuple(g_log), 
                       file_path = os.path.join(checkpoint_dir, 'G_{}.ckpt'.format(step_i)))
            save_model(model = D, optimizer = D_optim, step = step_i, log = tuple(d_log), 
                       file_path = os.path.join(checkpoint_dir, 'D_{}.ckpt'.format(step_i)))

            plot_loss(g_log = g_log, d_log = d_log, file_path = os.path.join(checkpoint_dir, 'loss.png'))
            plot_classifier_loss(log = classifier_log, file_path = os.path.join(checkpoint_dir, 'classifier loss.png'))

            generation_by_attributes(model = G, device = args.device, step = step_i, latent_dim = latent_dim, 
                                     hair_classes = num_classes,
                                     sample_dir = fixed_attribute_dir)
    
if __name__ == '__main__':
    main()
            
