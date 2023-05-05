# -*- coding: utf-8 -*-
'''
Adversarial Attack on Classification Models (ResNets)

This program is intended to be a general tool for training any classification
model and test its robustness with any attack strategy. To approach other 
models just changes the 'model.py' file, or load the desired model in the va-
riable 'model'. And to change the attack, just change the attack class 'atk'
with the desired attack strategy (the hint is to use Cleverhans or Torchattacks
libraries).

This algorithm uses Cleverhans library to attack a pre-trained model of 99.29%
accuracy in the classification of the white-blood cell types on blood stained
slides. Using the Cleverhans library, a diverse range of attacks can be
conducted using this same file, only changing the class called at the 'atk'
object to the desired attack method. See Cleverhans documentation for more
information.

This program can only run with other three python files (utils.py, model.py and
dataset.py), and with a trained model, which can be trained using the train.py
file. It is interesting to train the model with the same dataset used in the
attacks. Thus, specify the path for this dataset in the list variables in this
algorithm, e.g. 'train_image_dir' (this is the same name used in train.py for
training). All python files are available on GitHub repository (link below).

- 'utils.py': used to define util functions both for attacking and training.
- 'dataset.py': to load images in the DataLoader class of Torch.
- 'model.py': defined the model itself (it can be changed to any desired model)
which in this case can be either ResNets with 18, 34, 50, 101, and 152 layers.
- 'train.py': algorithm that can be used for training any model described in
'model.py'

To attack your mode in the white-blood cells problem, you need to train it on
WBC datasets, like the Raabin-WBC Dataset (used here), using the 'train.py'
file, or on other dataset of your choise.

P.S.: the cell with main function has to be executed separately in the python
console (copy and paste this cell in the console separately)

Find more on the GitHub Repository:
https://github.com/MarlonGarcia/attacking-white-blood-cells

@author: Marlon Rodrigues Garcia
@instit: University of São Paulo
'''

### Program Header

# importing libraries
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import cv2
import pandas as pd
# if running on Colabs, mounting drive
run_on_colabs = False
if run_on_colabs:
    # importing Drive
    from google.colab import drive
    drive.mount('/content/gdrive')
    # to import add current folder to path (import py files):
    import sys
    root_folder = '/content/gdrive/MyDrive/College/Biophotonics Lab/Research/Programs/Python/Adversarial Attacks/attacking-white-blood-cells/attacking-white-blood-cells/Classification'
    sys.path.append(root_folder)
else:
    root_folder = 'C:/Users/marlo/My Drive/College/Biophotonics Lab/Research/Programs/Python/Adversarial Attacks/attacking-white-blood-cells/attacking-white-blood-cells/Classification'
import os
os.chdir(root_folder)
from utils import *
from model import ResNet50
from absl import flags
from cleverhans.torch.attacks.projected_gradient_descent import (
    projected_gradient_descent
)


#%% Defining Hyperparameters and Directories

# hyperparameters
epsilons = [0, 8/255, 16/255, 32/255, 64/255, 128/255]
device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 1          # batch size
learning_rate = 1e-3    # defining learning rate
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

# defining the path to datasets
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
else:
    load_model_dir = 'C:/Users/marlo/My Drive/College/Biophotonics Lab/Research/Programs/Python/Camera & Image/Microscópio Veterinário/Atuais/ResNet/backup/2022.10.20 - ResNet50, 60epoc, 40batch, Raabin'



#%% Defining Model and Loading DataLoader 

## The next parameters have to be defined before loading the model
# defining the UResNet50 from 'model.py' and casting to device
model = ResNet50(in_channels=3, num_classes=5).to(device)
# defining the loss function to be used
loss_fn = nn.CrossEntropyLoss()
# defining the optimizer (there are commented options below)
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
# defining optimizer scheduling
schedule = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

# loading the pre-trained weights for the ResNet50
if load_model:
    # loading checkpoint, if 'cpu', we need to pass 'map_location'
    os.chdir(load_model_dir)
    if device == 'cuda':
        load_checkpoint(torch.load('my_checkpoint61.pth.tar'), model)
    else:
        load_checkpoint(torch.load('my_checkpoint61.pth.tar',
                                   map_location=torch.device('cpu')), model)

# loading DataLoaders for testing and validating datasets
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
# loading DataLoaders for dataset to save images (sample dataset)
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

#%% Running  the main() function, where the model will be attacked and evaluated


def norm255(image):
    # function to normalize between 0 and 255
    image = image - np.min(image)
    image = image/(np.max(image)/255)
    return np.array(image, np.uint8)


def norm255tensor(image):
    # function to normalize a tensor between 0 and 255
    image = image - image.min()
    image = image/(image.max()/255)
    return image.to(torch.uint8)


def attack(model, loader, epsilon):
    # defining the attack function
    # setting the model to evaluation
    model.eval()
    # starting the metrics with zero
    num_correct = 0
    # using 'tqdm' library to see a progress bar
    loop = tqdm(loader, desc='Check acc')
    for i, (dictionary, label) in enumerate(loop):
        # finding image from dictionary
        image = dictionary['image0']
        # label has to be in 'LongTensor' type
        label = label.type(torch.LongTensor)
        # casting to device
        image, label = image.to(device), label.to(device)
        # normalizing to enter in 'atk' instance
        min_val = image.min()
        image1 = image - min_val
        max_val = image1.max()
        image1 = image1/max_val
        # calculating attack from input image
        image_adv = projected_gradient_descent(model, image1, epsilon,
                                               0.01, 40, np.inf)
        # de-normalizing to the standard range (model only understand the
        # standard range, with zero mean and unitary standard deviation)
        image_adv = image_adv*max_val
        image_adv = image_adv+min_val
        # forward-passing the adversarial example
        output = model(image_adv)
        # summing the currect classificatios
        num_correct += (output.argmax(1)==label).type(torch.float).sum().item()
        # showing accuracy in the progress bar with 4 decimals
        loop.set_postfix(acc=str(round(100*num_correct/len(loader.dataset),4)))
    # returning accuracy
    return 100*num_correct/len(loader.dataset)


def label_name(label):
    ''' Function to find the label name from it python's number
    
    'label' (int or np.array): label integer number
    return (string): related label name
    '''
    # defining a table of correlation
    names = ['Basophil', 'Eosinophil', 'Lymphocyte', 'Monocyte', 'Neutrophil']
    # returning name
    return names[int(label)]


def tensor2img(tensor):
    # function to convert a tensor to a printable image
    return tensor.detach().permute(0,2,3,1).to('cpu').numpy()[0,:,:,:]


#%% Defining and running the main function (this cell has to be executed in the
# console separately, so copy and paste in the console)

def main(model, epsilons, test_loader, valid_loader, save_loader, root_folder):
    # defining lists to add the accuracies found
    accuracy_test = []
    accuracy_valid = []
    # iterating in 'epsilons' to accounts for different epsilon in perturbation
    for epsilon in epsilons:
        # setting the model to evaluation
        model.eval()
        
        # first, attacking the validation dataset
        print('\n- Attacking the Validation Dataset...\n')
        acc = attack(model, test_loader, epsilon)
        # saving accuracy
        accuracy_valid.append(acc)
        print('\nAccuracy:', round(acc,2), ', for Epsilon:', round(epsilon,3))
        
        # then, attacking the testing dataset
        print('\n- Attacking the Testing Dataset...\n')
        acc = attack(model, valid_loader, epsilon)
        # saving accuracy
        accuracy_test.append(acc)
        print('\nAccuracy:', round(acc,2), ', for Epsilon:', round(epsilon,3))
        
        ## Saving oroginal and perturbed images.
        os.chdir(root_folder)
        # creating directory to save results
        os.chdir(root_folder)
        try: os.mkdir('save_images')
        except: pass
        # setting the model to evaluation
        model.eval()
        for i, (dictionary, label) in enumerate(save_loader):
            # finding image from dictionary
            image = dictionary['image0']
            # label has to be in 'LongTensor' type
            label = label.type(torch.LongTensor)
            # casting to device
            image, label = image.to(device), label.to(device)
            # normalizing to enter in 'atk' instance
            min_val = image.min()
            image1 = image - min_val
            max_val = image1.max()
            image1 = image1/max_val
            # calculating attack from input image
            image_adv = projected_gradient_descent(model, image1, epsilon,
                                                   0.01, 40, np.inf)
            # de-normalizing to the standard range (model only understand the
            # standard range, with zero mean and unitary standard deviation)
            image_adv = image_adv*max_val
            image_adv = image_adv+min_val
            # forward-passing the adversarial example
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
    main(model, epsilons, test_loader, valid_loader, save_loader, root_folder)

#%% Saving Data (csv) and Plotting Accuracies

# saving data using Pandas library
dictionary = {'acc-valid': accuracy_valid, 'acc-test': accuracy_test}
df = pd.DataFrame(dictionary, columns = ['acc-valid', 'acc-test'])
os.chdir(root_folder)
df.to_csv('dictionary.csv', index = False)
    

# plotting image of acuraccies and dice-score
plt.subplots()
plt.plot(epsilons, accuracy_valid, label='accuracies valid.')
plt.plot(epsilons, accuracy_test, label='accuracies test')
plt.xlabel('Perturbations\' $\epsilon$')
plt.ylabel('Accuracy and Dice-Score')
plt.legend()
