import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from torchvision.utils import save_image

# Number to text mapping of class labels
hair_mapping =  ['1', '2', '3', '4', '5', '6', '7', 
                 '8', '9', '0']



def denorm(img):
    """ Denormalize input image tensor. (From [0,1] -> [-1,1]) 
    
    Args:
        img: input image tensor.
    """
	
    output = img / 2 + 0.5
    return output.clamp(0, 1)

def save_model(model, optimizer, step, log, file_path):
    """ Save model checkpoints. """

    state = {'model' : model.state_dict(),
             'optim' : optimizer.state_dict(),
             'step' : step,
             'log' : log}
    torch.save(state, file_path)
    return

def load_model(model, optimizer, file_path):
    """ Load previous checkpoints. """

    prev_state = torch.load(file_path)
    
    model.load_state_dict(prev_state['model'])
    optimizer.load_state_dict(prev_state['optim'])
    start_step = prev_state['step']
    log = prev_state['log']
    
    return model, optimizer, start_step, log
    
def show_process(total_steps, step_i, g_log, d_log, classifier_log):
    """ Show relevant losses during training. """

    print('Step {}/{}: G_loss [{:8f}], D_loss [{:8f}], Classifier loss [{:8f}]'.format(
            step_i, total_steps, g_log[-1], d_log[-1], classifier_log[-1]))
    return

def plot_loss(g_log, d_log, file_path):
    """ Plot generator and discriminator losses. """

    steps = list(range(len(g_log)))
    plt.semilogy(steps, g_log)
    plt.semilogy(steps, d_log)
    plt.legend(['Generator Loss', 'Discriminator Loss'])
    plt.title("Loss ({} steps)".format(len(steps)))
    plt.savefig(file_path)
    plt.close()
    return

def plot_classifier_loss(log, file_path):
    """ Plot auxiliary classifier loss. """
    
    steps = list(range(len(log)))
    plt.semilogy(steps, log)
    plt.legend(['Classifier Loss'])
    plt.title("Classifier Loss ({} steps)".format(len(steps)))
    plt.savefig(file_path)
    plt.close()
    return

def get_random_label(batch_size, hair_classes, hair_prior, eye_classes, eye_prior):
    """ Sample a batch of random class labels given the class priors.
    
    Args:
        batch_size: number of labels to sample.
        hair_classes: number of hair colors. 
        hair_prior: a list of floating points values indicating the distribution
					      of the hair color in the training data.
        eye_classes: (similar as above).
        eye_prior: (similar as above).
    
    Returns:
        A tensor of size N * (hair_classes + eye_classes). 
    """
    
    hair_code = torch.zeros(batch_size, hair_classes)  # One hot encoding for hair class
  

    hair_type = np.random.choice(hair_classes, batch_size, p = hair_prior)  # Sample hair class from hair class prior
    
    
    for i in range(batch_size):
        hair_code[i][hair_type[i]] = 1
       

    return hair_code

def generation_by_attributes(model, device, latent_dim, hair_classes,
    sample_dir, step = None, fix_hair = None, fix_eye = None):
    """ Generate image samples with fixed attributes.
    
    Args:
        model: model to generate images.
        device: device to run model on.
        step: current training step. 
        latent_dim: dimension of the noise vector.
        fix_hair: Choose particular hair class. 
                  If None, then hair class is chosen randomly.
        hair_classes: number of hair colors.
        fix_eye: Choose particular eye class. 
                 If None, then eye class is chosen randomly.
        eye_classes: number of eye colors.
        sample_dir: folder to save images.
    
    Returns:
        None
    """
    
    hair_tag = torch.zeros(64, hair_classes).to(device)
   
    hair_class = np.random.randint(hair_classes)

    for i in range(64):
    	hair_tag[i][hair_class] = 1

    
    z = torch.randn(64, latent_dim).to(device)

    output = model(z, hair_tag)
    if step is not None:
        file_path = '{} hair , step {}.png'.format(hair_mapping[hair_class], step)
    else:
        file_path = '{} hair .png'.format(hair_mapping[hair_class])
    save_image(denorm(output), os.path.join(sample_dir, file_path))




