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
'''

### Program Header

# To use Google Colabs, choose 'True' for 'colabs'
colabs = False

if colabs:
    # Importing Drive
    from google.colab import drive
    drive.mount('/content/gdrive')
    # To import add current folder to path (import py files):
    import sys
    root_folder = '/content/gdrive/MyDrive/College/Biophotonics Lab/Research/Programs/Python/Adversarial Attacks'
    save_results_dir = '/content/gdrive/MyDrive/College/Biophotonics Lab/Research/Programs/Python/Adversarial Attacks'
    root_model = '/content/gdrive/MyDrive/College/Biophotonics Lab/Research/Programs/Python/Camera & Image/Microscópio Veterinário/Atuais/ResNet em U/saved_models/UResNet50 - Adam batch 8 lr standard Adam'
    sys.path.append(root_folder)
else:
    root_folder = 'C:/Users/marlo/My Drive/College/Biophotonics Lab/Research/Programs/Python/Adversarial Attacks'
    save_results_dir = 'C:/Users/marlo/My Drive/College/Biophotonics Lab/Research/Programs/Python/Adversarial Attacks'
    root_model = 'C:/Users/marlo/My Drive/College/Biophotonics Lab/Research/Programs/Python/Camera & Image/Microscópio Veterinário/Atuais/ResNet em U/saved_models/UResNet50 - Adam batch 8 lr standard Adam'

# Importing Libraries
import os
os.chdir(root_folder)
from utils import *
from tqdm import tqdm
import torch
import torchattacks
import torch.nn as nn
import torchvision.transforms.functional as tf
from torchmetrics import Dice
import matplotlib.pyplot as plt
import numpy as np
import cv2
import pandas as pd
from model import UResNet50



#%% Defining Hyperparameters and Directories

# Hyperparameters
epsilons = [0, .05, .1, .15, .2, .25, .3]
device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 4          # batch size
num_workers = 2         # number of workers (smaller or = n° processing units)
clip_train = 1.00       # percentage to clip the train dataset (for tests)
clip_valid = 1.00       # percentage to clip the valid dataset (for tests)
valid_percent = 0.15    # use a percent of train dataset as validation dataset
test_percent = 0.15     # a percent from training dataset (but do not excluded)
image_height = 224      # height to crop the image
image_width = 224       # width to crop the image
pin_memory = True       # choose to pin memory
load_model = True       # 'true' to load a model and test it, or use it
save_images = True      # 'true' to save model trained after epoches

# Images' Directories for the datasets
if colabs:
    # Images' Directories to be used in Colabs
    train_image_dir = ['/content/gdrive/Shareddrives/Veterinary Microscope/Dataset/Raabin-WBC Data - Nucleus_cytoplasm_Ground truths/GrTh/Original/Basophil',
                        '/content/gdrive/Shareddrives/Veterinary Microscope/Dataset/Raabin-WBC Data - Nucleus_cytoplasm_Ground truths/GrTh/Original/Eosinophil',
                        '/content/gdrive/Shareddrives/Veterinary Microscope/Dataset/Raabin-WBC Data - Nucleus_cytoplasm_Ground truths/GrTh/Original/Lymphocyte',
                        '/content/gdrive/Shareddrives/Veterinary Microscope/Dataset/Raabin-WBC Data - Nucleus_cytoplasm_Ground truths/GrTh/Original/Monocyte',
                        '/content/gdrive/Shareddrives/Veterinary Microscope/Dataset/Raabin-WBC Data - Nucleus_cytoplasm_Ground truths/GrTh/Original/Neutrophil']
    # Directory with images to save
    save_image_dir = ['/content/gdrive/Shareddrives/Veterinary Microscope/Dataset/Raabin-WBC Data - Nucleus_cytoplasm_Ground truths/GrTh/Original/image']

else:
    # Images' Directories to be used on windows desktop
    train_image_dir = ['G:/Shared drives/Veterinary Microscope/Dataset/Raabin-WBC Data - Nucleus_cytoplasm_Ground truths/GrTh/Original/Basophil',
                       'G:/Shared drives/Veterinary Microscope/Dataset/Raabin-WBC Data - Nucleus_cytoplasm_Ground truths/GrTh/Original/Eosinophil',
                       'G:/Shared drives/Veterinary Microscope/Dataset/Raabin-WBC Data - Nucleus_cytoplasm_Ground truths/GrTh/Original/Lymphocyte',
                       'G:/Shared drives/Veterinary Microscope/Dataset/Raabin-WBC Data - Nucleus_cytoplasm_Ground truths/GrTh/Original/Monocyte',
                       'G:/Shared drives/Veterinary Microscope/Dataset/Raabin-WBC Data - Nucleus_cytoplasm_Ground truths/GrTh/Original/Neutrophil']
    # Directory with images to save
    save_image_dir = ['G:/Shared drives/Veterinary Microscope/Dataset/Raabin-WBC Data - Nucleus_cytoplasm_Ground truths/GrTh/Original/image']

val_image_dir = None


#%% Defining Model and Loading DataLoader 

# Defining the UResNet50 from 'model.py' file
model = UResNet50(in_channels=3, num_classes=3).to(device)

# Loading the Pre-trained weights for the UResNet50
if load_model:
    # Loading checkpoint, if 'cpu', we need to pass 'map_location'
    os.chdir(root_model)
    if device == 'cuda':
        load_checkpoint(torch.load('my_checkpoint80.pth.tar'), model)
    else:
        load_checkpoint(torch.load('my_checkpoint80.pth.tar',
                                   map_location=torch.device('cpu')), model)

# Defining the loss function to be used
loss_fn = nn.CrossEntropyLoss()

# Loading DataLoaders
_, test_loader, valid_loader = get_loaders(
    train_image_dir=train_image_dir,
    valid_percent=valid_percent,
    test_percent=test_percent,
    batch_size=batch_size,
    image_height=image_height,
    image_width=image_width,
    num_workers=num_workers,
    pin_memory=pin_memory,
    val_image_dir=val_image_dir,
    clip_valid=clip_valid,
    clip_train=clip_train
)

#%% Running main() function, where the model will be attacked and evaluated

accuracy_test = []
dice_score_test = []
accuracy_valid = []
dice_score_valid = []

def norm255(image):
    image = image - np.min(image)
    image = image/(np.max(image)/255)
    return np.array(image, np.uint8)

def main():
    # Iterating in 'epsilons' to accounts for different epsilon in perturbation
    for epsilon in epsilons:
        # Setting the model to evaluation
        model.eval()
        # Defining the attack to Projected Gradient Method (PGD)
        atk = torchattacks.PGD(model, eps=epsilon, alpha=2/255, steps=10)
        # Setting the mode to choose the label of attack
        atk.set_mode_targeted_by_label()
        ## First, the attack will be focused in the 'Test Dataset'
        # Starting the metrics with zero
        num_correct = 0
        num_pixels = 0
        dice_scores = 0
        # Using 'tqdm' library to see a progress bar
        loop = tqdm(test_loader)
        for i, dictionary in enumerate(loop):
            # Finding the dictionary keys for image and label
            image, label = dictionary
            # Extracting the data from dictionary with keys 'image' and 'label'
            x, y = dictionary[image], dictionary[label]
            # Casting the data to device
            x, y = x.to(device=device), y.to(device=device)
            # Label needs to be float for comparison with the output
            y = y.float()
            # Choosing ones to the adversarial label (one is label of cytoplasm)
            label_adv = torch.Tensor(np.ones(np.shape(y),float))
            # The next steps are to normalize the image to [0,1] (atk needs it)
            min_x = x.min()
            x1 = x - min_x
            max_x = x1.max()
            x1 = x1/max_x
            # Calculating attack from input image and the adversarial label
            x_adv = atk(x1, label_adv)
            # De-normalizing to the standard range (model only understand the
            # standard range, with zero mean and unitary standard deviation)
            x_adv = x_adv*max_x
            x_adv = x_adv+min_x
            # Forward-passing the adversarial example
            output = model(x_adv)
            # Cropping the label, in case the convolutions changed output size
            y = tf.center_crop(y, output.shape[2:])
            # Transforming predictions in 'True' or 'False', than in float
            output = (output > 0.5).float()
            # Calculating the number of currectly-predicted pixels
            num_correct += (output==y).sum()
            # Calculating the number of pixels to statistics
            num_pixels += torch.numel(output)
            ## Calculating dice-score (data needs to be integer)
            output = output.to(device='cpu').to(torch.int32)
            y = y.to(device='cpu').to(torch.int32)
            # Defining which index to be ignored in the dice-score calculation,
            # which means defining which label is tha background (is the zero)
            dice = Dice(ignore_index=0)
            # Calculating the dice-score
            dice_scores += dice(output,y)
            # Next statement is to save images from 5 first items (label and
            # perturbed image)
            if save_images and i<5:
                os.chdir(root_folder)
                # Tensor needs to turn into numpy (with appropriate format)
                temp = y.permute(0,2,3,1).to('cpu',torch.int32).numpy()[0,:,:,:]
                # Tensor is zero mean and unitary value, so we use 'norm255'
                # to de-normalize the tensor to the image range (0 to 255)
                temp = norm255(temp)
                # Saving image with OpenCV
                cv2.imwrite('saved_images/Eps'+str(epsilon)+'_y_test_'+str(i)+'.png', temp)
                # Transforming tensor to numpy with appropriate format
                temp = output.permute(0,2,3,1).to('cpu',torch.int32).numpy()[0,:,:,:]
                # De-normalizing tensor to the range of an image
                temp = norm255(temp)
                # Saving the image
                cv2.imwrite('saved_images/Eps'+str(epsilon)+'_y_pert_test_'+str(i)+'.png', temp)
        
        # Calculating accuracy and dice-score, and appending them
        acc = 100*num_correct.item()/num_pixels
        dice_item = 100*dice_scores.item()/len(test_loader)
        accuracy_test.append(acc)
        dice_score_test.append(dice_item)
        print('Accuracy:', acc, ', for Epsilon:', epsilon)
        
        ## Now, the attack will be focused in the Validation Dataset. The same
        # comments found in the above statement, can meet the bellow iteration
        num_correct = 0
        num_pixels = 0
        dice_scores = 0
        loop = tqdm(valid_loader)
        for i, dictionary in enumerate(loop):
            # finding the dictionary keys for image and label
            image, label = dictionary
            # extracting the data from dictionary with keys 'image' and 'label'
            x, y = dictionary[image], dictionary[label]
            x, y = x.to(device=device), y.to(device=device)
            y = y.float()
            label_adv = torch.Tensor(np.ones(np.shape(y),float))
            min_x = x.min()
            x1 = x - min_x
            max_x = x1.max()
            x1 = x1/max_x
            x_adv = atk(x1, label_adv)
            x_adv = x_adv*max_x
            x_adv = x_adv+min_x
            output = model(x_adv)
            y = tf.center_crop(y, output.shape[2:])
            output = (output > 0.5).float()
            num_correct += (output==y).sum()
            num_pixels += torch.numel(output)
            output = output.to(device='cpu').to(torch.int32)
            y = y.to(device='cpu').to(torch.int32)
            dice = Dice(ignore_index=0)
            dice_scores += dice(output,y)
            if save_images and i<5:
                os.chdir(root_folder)
                temp = y.permute(0,2,3,1).to('cpu').numpy()[0,:,:,:]
                temp = norm255(temp)
                cv2.imwrite('saved_images/Eps'+str(epsilon)+'_y_valid_'+str(i)+'.png', temp)
                temp = output.permute(0,2,3,1).to('cpu').numpy()[0,:,:,:]
                temp = norm255(temp)
                cv2.imwrite('saved_images/Eps'+str(epsilon)+'_y_pert_valid_'+str(i)+'.png', temp)
            
        acc = 100*num_correct.item()/num_pixels
        dice_item = 100*dice_scores.item()/len(valid_loader)
        accuracy_valid.append(acc)
        dice_score_valid.append(dice_item)
        print('Accuracy:', acc, ', for Epsilon:', epsilon)
        
        ## Saving images - The next 'get_loaders' and the next 'for' loop in
        #'save_loader' are used to save original and perturbed images. This
        # images cannot be saved in the above loops, because the 'get_loaders'
        # lose part of the image informatino due to its normalization
        print('- Saving Images...')
        # Loading the DataLoader
        save_loader, _, _ = get_loaders(
            train_image_dir=save_image_dir,
            valid_percent=0,
            test_percent=0,
            batch_size=1,
            image_height=image_height,
            image_width=image_width,
            num_workers=num_workers,
            pin_memory=pin_memory,
            val_image_dir=None,
            clip_valid=1.0,
            clip_train=1.0
        )
        # Defining names of images to be saved (important for original ones)
        names = os.listdir(save_image_dir[0])
        for i, dictionary in enumerate(save_loader):
            image, label = dictionary
            # Extracting the data from dictionary with keys 'image' and 'label'
            x, y = dictionary[image], dictionary[label]
            # Casting to device
            x, y = x.to(device=device), y.to(device=device)
            # Label needs to be float
            y = y.float()
            # Criating one adversarial label to follow (label equal to one
            # Means the whole image is cytoplasm)
            label_adv = torch.Tensor(np.ones(np.shape(y),float))
            # Next steps are to normalize x to [0,1], needed for 'atk'
            min_x = x.min()
            x1 = x - min_x
            max_x = x1.max()
            x1 = x1/max_x
            # Creating one attack based on a label of ones
            x_adv = atk(x1, label_adv)
            # Next steps are to de-normalize to a value scale that the model
            # understand
            x_adv = x_adv*max_x
            x_adv = x_adv+min_x
            # Forward-passing the perturbed image
            output = model(x_adv)
            # Crop the label, in case the convolutions changed the image size
            y = tf.center_crop(y, output.shape[2:])
            # Turning probabilities in 'True' or 'False', then in float values
            output = (output > 0.5).float()
            ## Next lines are to actually save the images
            os.chdir(root_folder)
            # The tensor needs to be converted to numpy to be saved
            temp = y.permute(0,2,3,1).to('cpu',torch.int32).numpy()[0,:,:,:]
            # Function to de-normalize the tensor to the [0,255] range
            temp = norm255(temp)
            # Saving image with OpenCV
            cv2.imwrite('save_images/Eps'+str(epsilon)+'_y_'+str(i)+'.png', temp)
            # Converting tensor to numpy in appropriate form
            temp = output.permute(0,2,3,1).to('cpu',torch.int32).numpy()[0,:,:,:]
            # De-normalizing the values
            temp = norm255(temp)
            # Saving image with OpenCV
            cv2.imwrite('save_images/Eps'+str(epsilon)+'_y_pert_'+str(i)+'.png', temp)
            ## The next steps are to load (imread) and save (imwrite) the ori-
            # ginal and preturbed images
            temp = cv2.imread(save_image_dir[0]+'/'+names[i])
            # Next steps are to return the numpy to the tensor format 
            temp = temp.astype(float)/np.max(temp)
            temp = torch.Tensor(temp).permute(2,0,1)
            temp = temp[None,:,:,:]
            temp = tf.resize(temp,(image_height,image_width))
            # Attacking the original image
            temp_adv = atk(temp, label_adv)
            # Returning to the numpy format to be saved as an image
            temp = temp.permute(0,2,3,1).to('cpu').numpy()[0,:,:,:]
            # De-normalization to the image range
            temp = norm255(temp)
            # Saving image
            cv2.imwrite('save_images/Eps'+str(epsilon)+'_x_'+str(i)+'.png', temp)
            # Returning from tensor to desired format to save
            temp_adv = temp_adv.permute(0,2,3,1).to('cpu').numpy()[0,:,:,:]
            # De-normalization to the [0,255] range
            temp_adv = norm255(temp_adv)
            # Saving image
            cv2.imwrite('save_images/Eps'+str(epsilon)+'_x_pert_'+str(i)+'.png', temp_adv)
            

if __name__ == '__main__':
    main()

#%% Saving Data (csv) and Plotting Accuracies

# Saving Data using Pandas Library
dictionary = {'acc-valid': accuracy_valid, 'dice-valid': dice_score_valid,
              'acc-test': accuracy_test, 'dice-test': dice_score_test}
df = pd.DataFrame(dictionary, columns = ['acc-valid', 'dice-valid',
                                         'acc-test', 'dice-test'])
os.chdir(root_folder)
df.to_csv('dictionary.csv', index = False)


# Plotting Image of Acuraccies and Dice-Score
plt.subplots()
plt.plot(epsilons, accuracy_valid, label='accuracies valid.')
plt.plot(epsilons, dice_score_valid, label='dice-score valid.')
plt.plot(epsilons, accuracy_test, label='accuracies test')
plt.plot(epsilons, dice_score_test, label='dice-score test')
plt.xlabel('Perturbations\' $\varepsilon$')
plt.ylabel('Accuracy and Dice-Score')
plt.legend()
