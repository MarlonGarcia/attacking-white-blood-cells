# -*- coding: utf-8 -*-
'''
Adversarial Attack on a UResNet50 Neural Network

This program uses Cleverhans library to attack a pre-trained model of 99.29%
accuracy in the classification of the white-blood cell types on blood stained
slides. Using the Cleverhans library, a diverse range of attacks can be
conducted using this same file, only changing the class called at the 'atk'
object to the desired attack method. See Cleverhans documentation for more
information.

This program can only run with other three python files, named 'utils.py', 
with the utils functions to be used here, 'dataset.py' to load the dataset
images from a torch DataLoader, and 'model.py' where the actual ResNet50
model can be found. There is also a supplementary 'train.py' file, which can
be used to conduct training of the UResNets (with 18, 34, 50, 101 or 152
layers). All python files are available on GitHub repository (link below)

To attack your model, you need to train it on the Raabin-WBC Dataset using the
'train.py' file, or on other dataset of your choise. This program can also be
the basis to develop automated attacks in other models and other datasets, just
by changing the model file to be loaded and the directories.

Find more on the GitHub Repository:
https://github.com/MarlonGarcia/attacking-white-blood-cells

@author: Marlon Rodrigues Garcia
@instit: University of São Paulo
'''

### Program Header

# Importing Libraries
from tqdm import tqdm
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import cv2
import pandas as pd
# If running on Colabs, mounting drive
run_on_colabs = False
if run_on_colabs:
    # Importing Drive
    from google.colab import drive
    drive.mount('/content/gdrive')
    # To import add current folder to path (import py files):
    import sys
    root_folder = '/content/gdrive/MyDrive/College/Biophotonics Lab/Research/Programs/Python/Adversarial Attacks/Classification'
    sys.path.append(root_folder)
else:
    root_folder = 'C:/Users/marlo/My Drive/College/Biophotonics Lab/Research/Programs/Python/Adversarial Attacks/Classification'
import os
os.chdir(root_folder)
from utils import *
from model import ResNet50
from absl import flags
from cleverhans.torch.attacks.projected_gradient_descent import (
    projected_gradient_descent
)


#% Defining Hyperparameters and Directories

# Hyperparameters
epsilons = [0, 8/255, 16/255, 32/255, 64/255, 128/255]
device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 1          # batch size
num_workers = 2         # number of workers (smaller or = n° processing units)
clip_train = 0.05       # percentage to clip the train dataset (for tests)
clip_valid = 0.05       # percentage to clip the valid dataset (for tests)
valid_percent = 0.15    # use a percent of train dataset as validation dataset
test_percent = 0.15     # a percent from training dataset (but do not excluded)
image_height = 300      # height to crop the image
image_width = 300       # width to crop the image
pin_memory = True       # choose to pin memory
load_model = True       # 'true' to load a model and test it, or use it
save_images = True      # 'true' to save model trained after epoches

# Defining the path to datasets
if run_on_colabs:
    train_image_dir = ['/content/gdrive/SharedDrives/Veterinary Microscope/Dataset/Raabin -WBC - Data Double-labeled Cropped Cells/Train2']
    val_image_dir = ['/content/gdrive/SharedDrives/Veterinary Microscope/Dataset/Raabin -WBC - Data Double-labeled Cropped Cells/Test2']
    save_image_dir = ['/content/gdrive/SharedDrives/Veterinary Microscope/Dataset/Raabin -WBC - Data Double-labeled Cropped Cells/images2save']
    csv_file_train =  ['/content/gdrive/SharedDrives/Veterinary Microscope/Dataset/Raabin -WBC - Data Double-labeled Cropped Cells/labels_train.csv']
    csv_file_valid =  ['/content/gdrive/SharedDrives/Veterinary Microscope/Dataset/Raabin -WBC - Data Double-labeled Cropped Cells/labels_test.csv']
    csv_file_save =  ['/content/gdrive/SharedDrives/Veterinary Microscope/Dataset/Raabin -WBC - Data Double-labeled Cropped Cells/labels_save.csv']
else:
    train_image_dir = ['G:/Shared drives/Veterinary Microscope/Dataset/Raabin -WBC - Data Double-labeled Cropped Cells/Train2']
    val_image_dir = ['G:/Shared drives/Veterinary Microscope/Dataset/Raabin -WBC - Data Double-labeled Cropped Cells/Test2']
    save_image_dir = ['G:/Shared drives/Veterinary Microscope/Dataset/Raabin -WBC - Data Double-labeled Cropped Cells/images2save']
    csv_file_train =  ['G:/Shared drives/Veterinary Microscope/Dataset/Raabin -WBC - Data Double-labeled Cropped Cells/labels_train.csv']
    csv_file_valid =  ['G:/Shared drives/Veterinary Microscope/Dataset/Raabin -WBC - Data Double-labeled Cropped Cells/labels_test.csv']
    csv_file_save =  ['G:/Shared drives/Veterinary Microscope/Dataset/Raabin -WBC - Data Double-labeled Cropped Cells/labels_save.csv']
# directory to save the results and to test the models:
if run_on_colabs:
    load_model_dir = '/content/gdrive/MyDrive/College/Biophotonics Lab/Research/Programs/Python/Camera & Image/Microscópio Veterinário/Atuais/ResNet/backup/2022.10.20 - ResNet50, 60epoc, 40batch, Raabin'
    save_results_dir = '/content/gdrive/MyDrive/College/Biophotonics Lab/Research/Programs/Python/Adversarial Attacks/Classification'
    test_models_dir = '/content/gdrive/MyDrive/College/Biophotonics Lab/Research/Programs/Python/Adversarial Attacks/Classification'
else:
    load_model_dir = 'C:/Users/marlo/My Drive/College/Biophotonics Lab/Research/Programs/Python/Camera & Image/Microscópio Veterinário/Atuais/ResNet/backup/2022.10.20 - ResNet50, 60epoc, 40batch, Raabin'
    save_results_dir = 'C:/Users/marlo/My Drive/College/Biophotonics Lab/Research/Programs/Python/Adversarial Attacks/Classification'
    test_models_dir = 'C:/Users/marlo/My Drive/College/Biophotonics Lab/Research/Programs/Python/Adversarial Attacks/Classification'



#% Defining Model and Loading DataLoader 

# Defining the UResNet50 from 'model.py' and casting to device
model = ResNet50(in_channels=3, num_classes=5).to(device)

# Loading the Pre-trained weights for the UResNet50
if load_model:
    # Loading checkpoint, if 'cpu', we need to pass 'map_location'
    os.chdir(load_model_dir)
    if device == 'cuda':
        load_checkpoint(torch.load('my_checkpoint61.pth.tar'), model)
    else:
        load_checkpoint(torch.load('my_checkpoint61.pth.tar',
                                   map_location=torch.device('cpu')), model)

# Defining the loss function to be used
loss_fn = nn.CrossEntropyLoss()

# Loading DataLoaders
_, test_loader, valid_loader = get_loaders(
    train_image_dir=train_image_dir,
    csv_file_train=csv_file_train,
    valid_percent=valid_percent,
    test_percent=test_percent,
    batch_size=batch_size,
    image_height=image_height,
    image_width=image_width,
    num_workers=num_workers,
    pin_memory=pin_memory,
    val_image_dir=None,
    csv_file_valid=None,
    clip_valid=clip_valid,
    clip_train=clip_train
)


#% Running main() function, where the model will be attacked and evaluated


def norm255(image):
    image = image - np.min(image)
    image = image/(np.max(image)/255)
    return np.array(image, np.uint8)


def norm255tensor(image):
    image = image - image.min()
    image = image/(image.max()/255)
    return image.to(torch.uint8)


def attack(model, loader, epsilon):
    # Setting the model to evaluation
    model.eval()
    # Starting the metrics with zero
    num_correct = 0
    # Using 'tqdm' library to see a progress bar
    loop = tqdm(loader, desc='Check acc')
    for i, (dictionary, label) in enumerate(loop):
        # Finding image from dictionary
        image = dictionary['image0']
        # Label has to be in 'LongTensor' type
        label = label.type(torch.LongTensor)
        # Casting to device
        image, label = image.to(device), label.to(device)
        # Normalizing to enter in 'atk' instance
        min_val = image.min()
        image1 = image - min_val
        max_val = image1.max()
        image1 = image1/max_val
        # Calculating attack from input image
        image_adv = projected_gradient_descent(model, image1, epsilon,
                                               0.01, 40, np.inf)
        # De-normalizing to the standard range (model only understand the
        # standard range, with zero mean and unitary standard deviation)
        image_adv = image_adv*max_val
        image_adv = image_adv+min_val
        # Forward-passing the adversarial example
        output = model(image_adv)
        # Summing the currect classificatios
        num_correct += (output.argmax(1)==label).type(torch.float).sum().item()
        # Showing accuracy in the progress bar with 4 decimals
        loop.set_postfix(acc=str(round(100*num_correct/len(loader.dataset),4)))
    # Returning accuracy
    return 100*num_correct/len(loader.dataset)


def label_name(label):
    ''' Function to find the label name from it python's number
    
    'label' (int or np.array): label integer number
    return (string): related label name
    '''
    # Defining a table of correlation
    names = ['Basophil', 'Eosinophil', 'Lymphocyte', 'Monocyte', 'Neutrophil']
    # Returning name
    return names[int(label)]


def tensor2img(tensor):
    # Function to convert a tensor to a printable image
    return tensor.detach().permute(0,2,3,1).to('cpu').numpy()[0,:,:,:]


#%%

accuracy_test = []
accuracy_valid = []

def main():
    # Iterating in 'epsilons' to accounts for different epsilon in perturbation
    for epsilon in epsilons:
        # Setting the model to evaluation
        model.eval()
        
        # First, attacking the Validation Dataset
        print('\n- Attacking the Validation Dataset...\n')
        acc = attack(model, test_loader, epsilon)
        # Saving accuracy
        accuracy_valid.append(acc)
        print('\nAccuracy:', round(acc,2), ', for Epsilon:', round(epsilon,3))
        
        # Then, attacking the Testing Dataset
        print('\n- Attacking the Testing Dataset...\n')
        acc = attack(model, valid_loader, epsilon)
        # Saving accuracy
        accuracy_test.append(acc)
        print('\nAccuracy:', round(acc,2), ', for Epsilon:', round(epsilon,3))
        
        ## Saving oroginal and perturbed images.
        os.chdir(root_folder)
        # Loading DataLoaders
        save_loader, _, _ = get_loaders(
            train_image_dir=save_image_dir,
            csv_file_train=csv_file_save,
            valid_percent=0,
            test_percent=0,
            batch_size=1,
            image_height=image_height,
            image_width=image_width,
            num_workers=num_workers,
            pin_memory=pin_memory,
            val_image_dir=None,
            csv_file_valid=None,
            clip_valid=1.0,
            clip_train=1.0
        )
        # Setting the model to evaluation
        model.eval()
        for i, (dictionary, label) in enumerate(save_loader):
            # Finding image from dictionary
            image = dictionary['image0']
            # Label has to be in 'LongTensor' type
            label = label.type(torch.LongTensor)
            # Casting to device
            image, label = image.to(device), label.to(device)
            # Normalizing to enter in 'atk' instance
            min_val = image.min()
            image1 = image - min_val
            max_val = image1.max()
            image1 = image1/max_val
            # Calculating attack from input image
            image_adv = projected_gradient_descent(model, image1, epsilon,
                                                   0.01, 40, np.inf)
            # De-normalizing to the standard range (model only understand the
            # standard range, with zero mean and unitary standard deviation)
            image_adv = image_adv*max_val
            image_adv = image_adv+min_val
            # Forward-passing the adversarial example
            output = model(image_adv)
            
            image = norm255tensor(image)
            image_adv = norm255tensor(image_adv)
            image_print = tensor2img(image)
            image_adv_print = tensor2img(image_adv)
                        
            im, ax = plt.subplots(1,2)
            ax[0].imshow(cv2.cvtColor(image_print,cv2.COLOR_BGR2RGB))
            ax[0].set_title('(a) '+label_name(label.item()))
            ax[0].axis('off')
            ax[1].imshow(cv2.cvtColor(image_adv_print,cv2.COLOR_BGR2RGB))
            ax[1].set_title('(b) '+label_name(output.argmax(1).item()))
            ax[1].axis('off')
            plt.pause(0.5)
            plt.tight_layout()
            plt.savefig('save_images/Eps'+str(round(epsilon,2))+'_image'+str(i)+'.png', bbox_inches='tight')
            plt.savefig('save_images/Eps'+str(round(epsilon,2))+'_image'+str(i)+'.svg', bbox_inches='tight')


if __name__ == '__main__':
    main()

#%% Saving Data (csv) and Plotting Accuracies

# Saving Data using Pandas Library
dictionary = {'acc-valid': accuracy_valid, 'acc-test': accuracy_test}
df = pd.DataFrame(dictionary, columns = ['acc-valid', 'acc-test'])
os.chdir(root_folder)
df.to_csv('dictionary.csv', index = False)
    

# Plotting Image of Acuraccies and Dice-Score
plt.subplots()
plt.plot(epsilons, accuracy_valid, label='accuracies valid.')
plt.plot(epsilons, accuracy_test, label='accuracies test')
plt.xlabel('Perturbations\' $\epsilon$')
plt.ylabel('Accuracy and Dice-Score')
plt.legend()
