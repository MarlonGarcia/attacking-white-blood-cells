# -*- coding: utf-8 -*-
'''
Adversarial Attack on Segmentation Models (UResNets)

This program is intended to be a general tool for training any segmentation
model and test its robustness with any attack strategy. To approach other 
models just changes the 'model.py' file, or load the desired model in the va-
riable 'model'. And to change the attack, just change the attack class 'atk'
with the desired attack strategy (the hint is to use Cleverhans or Torchattacks
libraries).

This algorithm uses Torchattacks library to attack a pre-trained model of 97%
accuracy in the segmentation of cytoplasm and nuclei regions in white-blood
cells (WBC) from blood stained slides. Using the Torchattacks library, a
diverse range of attacks can be conducted using this same algorithm, only by 
changing the class called at the 'atk' object to the desired attack method.
See Torchattacks documentation for more information.

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

# To use Google Colabs, choose 'True' for 'colabs'
colabs = False

if colabs:
    # importing Drive
    from google.colab import drive
    drive.mount('/content/gdrive')
    # to import add current folder to path (import py files):
    import sys
    root_folder = '/content/gdrive/MyDrive/College/Biophotonics Lab/Research/Programs/Python/Adversarial Attacks/Segmentation'
    root_model = '/content/gdrive/MyDrive/College/Biophotonics Lab/Research/Programs/Python/Camera & Image/Microscópio Veterinário/Atuais/ResNet em U/saved_models/UResNet50 - Adam batch 8 lr standard Adam'
    sys.path.append(root_folder)
else:
    root_folder = 'C:/Users/marlo/My Drive/College/Biophotonics Lab/Research/Programs/Python/Adversarial Attacks/Segmentation'
    root_model = 'C:/Users/marlo/My Drive/College/Biophotonics Lab/Research/Programs/Python/Camera & Image/Microscópio Veterinário/Atuais/ResNet em U/saved_models/UResNet50 - Adam batch 8 lr standard Adam'

# importing Libraries
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

# hyperparameters
epsilons = [8/255, 16/255, 32/255, 64/255, 128/255]
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

# images' Directories for the datasets
if colabs:
    # images' Directories to be used in Colabs
    train_image_dir = ['/content/gdrive/Shareddrives/Veterinary Microscope/Dataset/Raabin-WBC Data - Nucleus_cytoplasm_Ground truths/GrTh/Original/Basophil',
                        '/content/gdrive/Shareddrives/Veterinary Microscope/Dataset/Raabin-WBC Data - Nucleus_cytoplasm_Ground truths/GrTh/Original/Eosinophil',
                        '/content/gdrive/Shareddrives/Veterinary Microscope/Dataset/Raabin-WBC Data - Nucleus_cytoplasm_Ground truths/GrTh/Original/Lymphocyte',
                        '/content/gdrive/Shareddrives/Veterinary Microscope/Dataset/Raabin-WBC Data - Nucleus_cytoplasm_Ground truths/GrTh/Original/Monocyte',
                        '/content/gdrive/Shareddrives/Veterinary Microscope/Dataset/Raabin-WBC Data - Nucleus_cytoplasm_Ground truths/GrTh/Original/Neutrophil']
    # directory with images to save
    save_image_dir = ['/content/gdrive/Shareddrives/Veterinary Microscope/Dataset/Raabin-WBC Data - Nucleus_cytoplasm_Ground truths/GrTh/Original/image']

else:
    # images' Directories to be used on windows desktop
    train_image_dir = ['G:/Shared drives/Veterinary Microscope/Dataset/Raabin-WBC Data - Nucleus_cytoplasm_Ground truths/GrTh/Original/Basophil',
                       'G:/Shared drives/Veterinary Microscope/Dataset/Raabin-WBC Data - Nucleus_cytoplasm_Ground truths/GrTh/Original/Eosinophil',
                       'G:/Shared drives/Veterinary Microscope/Dataset/Raabin-WBC Data - Nucleus_cytoplasm_Ground truths/GrTh/Original/Lymphocyte',
                       'G:/Shared drives/Veterinary Microscope/Dataset/Raabin-WBC Data - Nucleus_cytoplasm_Ground truths/GrTh/Original/Monocyte',
                       'G:/Shared drives/Veterinary Microscope/Dataset/Raabin-WBC Data - Nucleus_cytoplasm_Ground truths/GrTh/Original/Neutrophil']
    # directory with images to save
    save_image_dir = ['G:/Shared drives/Veterinary Microscope/Dataset/Raabin-WBC Data - Nucleus_cytoplasm_Ground truths/GrTh/Original/image']

val_image_dir = None


#%% Defining Model and Loading DataLoader 

# defining the UResNet50 from 'model.py' file
model = UResNet50(in_channels=3, num_classes=3).to(device)

# loading the Pre-trained weights for the UResNet50
if load_model:
    # loading checkpoint, if 'cpu', we need to pass 'map_location'
    os.chdir(root_model)
    if device == 'cuda':
        load_checkpoint(torch.load('my_checkpoint80.pth.tar'), model)
    else:
        load_checkpoint(torch.load('my_checkpoint80.pth.tar',
                                   map_location=torch.device('cpu')), model)

# defining the loss function to be used
loss_fn = nn.CrossEntropyLoss()

# loading DataLoaders
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


# def one_hot2label(label):
#     output = torch.empty(label[:,0,:,:].squeeze().size())
#     output[label[:,1,:,:]==1] = 1
#     output[label[:,2,:,:]==1] = 2
#     return output


def main():
    # iterating in 'epsilons' to accounts for different epsilon in perturbation
    for epsilon in epsilons:
        # setting the model to evaluation
        model.eval()
        # defining the attack to Projected Gradient Method (PGD)
        atk = torchattacks.PGD(model, eps=epsilon, alpha=2/255, steps=10)
        # setting the mode to choose the label of attack
        atk.set_mode_targeted_by_label()
        # # setting the mode to random label
        # atk.set_mode_targeted_random()
        
        ## First, the attack will be focused in the Test Dataset
        # starting the metrics with zero
        num_correct = 0
        num_pixels = 0
        dice_scores = 0
        # using 'tqdm' library to see a progress bar
        loop = tqdm(test_loader)
        for i, dictionary in enumerate(loop):
            # finding the dictionary keys for image and label
            image, label = dictionary
            # extracting the data from dictionary with keys 'image' and 'label'
            x, y = dictionary[image], dictionary[label]
            # casting the data to device
            x, y = x.to(device=device), y.to(device=device)
            # label needs to be float for comparison with the output
            y = y.float()
            # changing background and cytoplasm labels (is one-hot enconding)
            label_adv = torch.Tensor(np.zeros(np.shape(y),float))
            label_adv[:,0,:,:][y[:,1,:,:]==1] = 1
            label_adv[:,1,:,:][y[:,2,:,:]==1] = 1
            label_adv[:,2,:,:][y[:,0,:,:]==1] = 1
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
                cv2.imwrite('saved_images/Eps'+str(epsilon)+'_output_pert_test_'+str(i)+'.png', temp)
        
        # Calculating accuracy and dice-score, and appending them
        acc = 100*num_correct.item()/num_pixels
        dice_item = 100*dice_scores.item()/len(test_loader)
        accuracy_test.append(acc)
        dice_score_test.append(dice_item)
        print('Accuracy:', acc, ', for Epsilon:', epsilon)
        
        ## Now, the attack will be focused in the Validation Dataset
        num_correct = 0
        num_pixels = 0
        dice_scores = 0
        loop = tqdm(valid_loader)
        for i, dictionary in enumerate(loop):
            # finding the dictionary keys for image and label
            image, label = dictionary
            # extracting the data from dictionary with keys 'image' and 'label'
            x, y = dictionary[image], dictionary[label]
            # Casting image and label to the device
            x, y = x.to(device=device), y.to(device=device)
            # Label has to be float type to be compared with output
            y = y.float()
            # Changing background and cytoplasm labels (is one-hot enconding)
            label_adv = torch.Tensor(np.zeros(np.shape(y),float))
            label_adv[:,0,:,:][y[:,1,:,:]==1] = 1
            label_adv[:,1,:,:][y[:,2,:,:]==1] = 1
            label_adv[:,2,:,:][y[:,0,:,:]==1] = 1
            # Next lines normalize 'x' to be used in 'atk' as 'x1'
            min_x = x.min()
            x1 = x - min_x
            max_x = x1.max()
            x1 = x1/max_x
            # Perturbing 'x1'
            x_adv = atk(x1, label_adv)
            # Returnig from normalization made above
            x_adv = x_adv*max_x
            x_adv = x_adv+min_x
            # Forward passing the adversarial example into the model
            output = model(x_adv)
            # Center crop label to 'output' size (in case of convolut. reduct.)
            y = tf.center_crop(y, output.shape[2:])
            # Transforming the output values in bytewise probability
            output = (output > 0.5).float()
            # Calculating number of currected classified pixels
            num_correct += (output==y).sum()
            # Calculating the number of pixels for statistics
            num_pixels += torch.numel(output)
            # Next lines are to calculate dice score
            output = output.to(device='cpu').to(torch.int32)
            y = y.to(device='cpu').to(torch.int32)
            dice = Dice(ignore_index=0)
            dice_scores += dice(output,y)
            # Saving label and perturbed label
            if save_images and i<5:
                os.chdir(root_folder)
                temp = y.permute(0,2,3,1).to('cpu').numpy()[0,:,:,:]
                temp = norm255(temp)
                cv2.imwrite('saved_images/Eps'+str(epsilon)+'_y_valid_'+str(i)+'.png', temp)
                temp = output.permute(0,2,3,1).to('cpu').numpy()[0,:,:,:]
                temp = norm255(temp)
                cv2.imwrite('saved_images/Eps'+str(epsilon)+'_output_pert_valid_'+str(i)+'.png', temp)
        
        # Acquiring  statistics
        acc = 100*num_correct.item()/num_pixels
        dice_item = 100*dice_scores.item()/len(valid_loader)
        accuracy_valid.append(acc)
        dice_score_valid.append(dice_item)
        print('Accuracy:', acc, ', for Epsilon:', epsilon)
        
        ## Saving images - The next 'get_loaders' and the next 'for' loop in
        #'save_loader' are used to save original and perturbed images. This
        # images cannot be saved in the above loops, because the 'get_loaders'
        # lose part of the image informatino due to its normalization
        print('- Saving Images...\n')
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
        # creating directory to save results
        os.chdir(root_folder)
        try: os.mkdir('save_images')
        except: pass
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
            # Changing background and cytoplasm labels (is one-hot enconding)
            label_adv = torch.Tensor(np.zeros(np.shape(y),float))
            label_adv[:,0,:,:][y[:,1,:,:]==1] = 1
            label_adv[:,1,:,:][y[:,2,:,:]==1] = 1
            label_adv[:,2,:,:][y[:,0,:,:]==1] = 1
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
            cv2.imwrite('save_images/Eps'+str(epsilon)+'_output_pert_'+str(i)+'.png', temp)
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
