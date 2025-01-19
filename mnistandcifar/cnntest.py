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


from utils2 import save_model, denorm, plot_loss, plot_classifier_loss, show_process
from utils2 import generation_by_attributes, get_random_label
import torch
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

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

# Define transformations for MNIST
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
num_classes = 10  # MNIST has 10 classes
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load test dataset
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Define a DataLoader for the test dataset
test_loader = DataLoader(dataset=test_dataset, batch_size=2, shuffle=False)

classifier_weights_path = '/samples/cnn/D_25000.ckpt'  # Replace with your path
prev_state = torch.load(classifier_weights_path)

D = Discriminator(10).to(device)
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
            class_predictions = discriminator(real_images)  # Assuming (real/fake, class predictions)
        
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
    plt.title("Confusion Matrix Only CNN")
   
    plt.savefig("confusion_matrix_ONLYCNN.jpg", dpi=300)
    plt.close()

    


# Evaluate the discriminator
evaluate_acgan_discriminator_classwise(D, test_loader, num_classes, device)