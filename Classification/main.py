# -*- coding: utf-8 -*-
'''
Adversarial Attack on a UResNet50 Neural Network

This program uses Torchattacks library to attack a pre-trained model of 97%
accuracy in segmenting cytoplasm and nuclei in white-blood cells on stained
blood slides. Using the torchattacks library, a diverse range of attacks can
be conducted using this same file, only changing how to call the 'atk' object
with the desired attack method. See 'torchattacks' documentation for more
information.

This program can only run with other three python files, named 'utils.py', 
with the utils functions to be used here, 'dataset.py' to load the dataset
images from a torch DataLoader, and 'model.py' where the actual UResNet50
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
import torchattacks
import torch.nn as nn
import torchvision.transforms.functional as tf
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


#% Defining Hyperparameters and Directories

# Hyperparameters
epsilons = [8/255, 16/255]
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


def attack(model, loader, atk):
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
            # Calculating attack from input image and the adversarial label
            image_adv = atk(image1, label)
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


#%%

accuracy_test = []
accuracy_valid = []

def main():
    # Iterating in 'epsilons' to accounts for different epsilon in perturbation
    for epsilon in epsilons:
        # Setting the model to evaluation
        model.eval()
        # Defining the attack to Projected Gradient Method (PGD)
        atk = torchattacks.PGD(model, eps=epsilon, alpha=2/255, steps=10)
        # # Setting the mode to choose the label of attack
        # atk.set_mode_targeted_by_label()
        # Setting the mode to random label
        atk.set_mode_targeted_random()
        
        # First, attacking the Validation Dataset
        print('\n- Attacking the Validation Dataset...\n')
        acc = attack(model, test_loader, atk)
        # Saving accuracy
        accuracy_valid.append(acc)
        print('\nAccuracy:', round(acc,2), ', for Epsilon:', round(epsilon,3))
        
        # Then, attacking the Testing Dataset
        print('\n- Attacking the Testing Dataset...\n')
        acc = attack(model, valid_loader, atk)
        # Saving accuracy
        accuracy_test.append(acc)
        print('\nAccuracy:', round(acc,2), ', for Epsilon:', round(epsilon,3))
        
        ## Saving oroginal and perturbed images.
        # This images cannot be saved in the above loops, because the
        # 'get_loaders' lose part of the image pre-processing
        print('\n- Saving Images...\n')
        # defining mean and standard deviation to normalization
        mean = [0.52096, 0.51698, 0.545980]
        std = [0.10380, 0.11190, 0.118877]
        # Defining labels to print
        labels = [4, 1, 1, 1, 4]
        # Defining names of images to be saved (important for original ones)
        names = os.listdir(save_image_dir[0])
        for i, (name, label) in enumerate(zip(names, labels)):
            image = cv2.imread(save_image_dir[0]+'/'+name)
            image = torch.from_numpy(image)
            image = image.permute(2,0,1)
            image = image.unsqueeze(0)
            image = tf.resize(image, (image_height, image_width))
            image = tf.normalize(image.float(), mean=mean, std=std)
            # Preparing label to enter 'atk'
            temp = torch.zeros(1,5)
            temp[0,int(label)] = 1
            label = temp.type(torch.LongTensor).to(device)
            # Casting to device
            image = image.to(device)
            # Normalizing to enter in 'atk' instance
            min_val = image.min()
            image1 = image - min_val
            max_val = image1.max()
            image1 = image1/max_val
            if True:
                print(model(image).size())
                print(model(image))
                print(label.size())
                print(label)
            # Calculating attack from input image and the adversarial label
            image_adv = atk(image1, label)
            # De-normalizing to the standard range (model only understand the
            # standard range, with zero mean and unitary standard deviation)
            image_adv = image_adv*max_val
            image_adv = image_adv+min_val
            # Forward-passing the adversarial example
            output = model(image_adv)
            # Preparing image to save (in uint8)
            image_adv = image_adv.to(torch.uint8)
            ## Next lines are to actually save the images
            os.chdir(root_folder)
            im, ax = plt.subplots(1,2)
            image = image.permute(1,2,0).to('cpu').numpy()
            image = norm255(image)
            ax[0].imshow(image)
            ax[0].axis('off')
            ax[0].set_title('(a)'+label_name(label))
            image_adv = image_adv.permute(1,2,0).to('cpu').numpy()
            image_adv = norm255(image_adv)
            ax[1].imshow(image_adv)
            ax[1].axis('off')
            ax[1].set_title('(b)'+label_name(output.argmax(1).item()))
            plt.tight_layout()
            plt.savefig('save_images/Eps'+str(epsilon)+'_image'+str(i)+'.png', bbox_inches='tight')


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
plt.xlabel('Perturbations\' $\varepsilon$')
plt.ylabel('Accuracy and Dice-Score')
plt.legend()