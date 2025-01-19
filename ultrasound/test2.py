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

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

import zipfile
import os
from PIL import Image
import numpy as np

from model3 import Generator, Discriminator
from utils3 import save_model, denorm, plot_loss, plot_classifier_loss, show_process
from utils3 import generation_by_attributes, get_random_label
import torch
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

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

# Output results
print(f"Loaded {len(images)} images with shape: {images[0].shape}")
print(f"Labels: {np.unique(labels)}")





# Instantiate the dataset
dataset = UltrasoundDataset(images [75:125], labels[75:125])
print(labels[75:125])

# Create a DataLoader
test_loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=4)

# Iterate through the DataLoader
for batch_idx, (images_batch, labels_batch) in enumerate(test_loader):
    print(f"Batch {batch_idx}:")
    print(f"Images shape: {images_batch.shape}")  # (batch_size, 3, H, W)
    print(f"Labels shape: {labels_batch.shape}")  # (batch_size,)
    break

device='cuda'
num_classes=2
# Define transformations for MNIST
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
# num_classes = 10  # MNIST has 10 classes
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# # Load test dataset
# test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# # Define a DataLoader for the test dataset
# test_loader = DataLoader(dataset=test_dataset, batch_size=2, shuffle=False)

classifier_weights_path = '/checkpoints/models1/D_40000.ckpt'  # Replace with your path
prev_state = torch.load(classifier_weights_path)

D = Discriminator(2).to(device)
D.load_state_dict(prev_state['model'])
D.eval()


def evaluate_acgan_discriminator_classwise(discriminator, test_loader, num_classes, device):
    """
    Evaluate the classification accuracy of the ACGAN discriminator on real test images, broken down by class.
    
    Args:
        discriminator: The ACGAN discriminator model.
        test_loader: DataLoader for the MNIST test dataset.
        num_classes: Number of classes (e.g., 10 for MNIST).
        device: The device (CPU or CUDA) to run the evaluation on.
    """
    discriminator.eval()  # Set discriminator to evaluation mode
    class_correct = [0] * num_classes  # Correct predictions per class
    class_total = [0] * num_classes  # Total samples per class
    all_real_labels = []
    all_predicted_labels = []
    with torch.no_grad():
        for real_images, real_labels in test_loader:
            # Move data to the device
            real_images, real_labels = real_images.to(device), real_labels.to(device)
            
            # Get discriminator outputs
            real_or_fake_score, class_predictions = discriminator(real_images)  # Assuming (real/fake, class predictions)
        
            # Predicted classes (argmax over class predictions)
            predicted_labels = class_predictions.argmax(dim=1)

                    # Store labels for confusion matrix
            all_real_labels.extend(real_labels.cpu().numpy())
            all_predicted_labels.extend(predicted_labels.cpu().numpy())
            
            # Update class-wise counters
            for i in range(num_classes):
                class_correct[i] += ((predicted_labels == i) & (real_labels == i)).sum().item()
                class_total[i] += (real_labels == i).sum().item()

    # Display class-wise accuracy
    for i in range(num_classes):
        accuracy = 100 * class_correct[i] / class_total[i] if class_total[i] > 0 else 0
        print(f"Class {i}: Accuracy = {accuracy:.2f}% ({class_correct[i]}/{class_total[i]} samples)")
  
    # Calculate and display overall accuracy
    total_correct = sum(class_correct)
    total_samples = sum(class_total)
    overall_accuracy = 100 * total_correct / total_samples
    print(f"Overall Accuracy: {overall_accuracy:.2f}%")
    # Display confusion matrix
    cm = confusion_matrix(all_real_labels, all_predicted_labels, labels=list(range(num_classes)))
    cmd = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(range(num_classes)))
    cmd.plot(cmap='Blues', xticks_rotation='vertical')
    plt.title("Confusion Matrix")
   
    plt.savefig("confusion_matrix.jpg", dpi=300)
    plt.close()

    


# Evaluate the discriminator
evaluate_acgan_discriminator_classwise(D, test_loader, num_classes, device)
