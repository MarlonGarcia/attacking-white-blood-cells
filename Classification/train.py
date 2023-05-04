# -*- coding: utf-8 -*-
"""
Program to Train, Save and Continue Training of ANN Models

This program is capable of training and saving models, also continue a training
from last epoch (use hyperparameter 'last_epoch' and 'continue_train=True', and
'load_model=True'), and to save resulting images (when used to images, bur it
is general porpouse), using 'save_images=True'.
                                                  
This program suports the optional use of a validation dataset, which directory
has to be passed through 'val_image_dir'. If this variable is 'None', then test
and validation datasets is splitted from train dataset using 'valid_percent'
and 'test_percent'. The test dataset is always clipped from the training data-
set.

In the case of fully-connected layers in the and, you can chose to change the
last fully-connected layer, adding one extra, just changing 'change_last_fc' to
True. It is also possible to test all saved models in the 'root_folder' direc-
tory by chosing 'test_models' to True.

Find more on the GitHub Repository:
https://github.com/MarlonGarcia/attacking-white-blood-cells

@author: Marlon Rodrigues Garcia
@instit: University of São Paulo
"""

### Program Header

import torch
import os
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
# if running on Colabs, mounting drive
run_on_colabs = False
if run_on_colabs:
    # importing Drive
    from google.colab import drive
    drive.mount('/content/gdrive')
    # to import add current folder to path (import py files):
    import sys
    root_folder = '/content/gdrive/MyDrive/College/Biophotonics Lab/Research/Programs/Python/Adversarial Attacks/Classification'
    sys.path.append(root_folder)
else:
    root_folder = 'C:/Users/marlo/My Drive/College/Biophotonics Lab/Research/Programs/Python/Adversarial Attacks/Classification'
import os
os.chdir(root_folder)
from utils import *
from model import ResNet50


#%% Defining Parameters and Path

# defining Hyperparameters
learning_rate = 1e-3    # learning rate
device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 16         # batch size
num_epochs = 62         # number of epochs
num_workers = 1         # number of workers
clip_train = 1          # percentage to clip the train dataset (for tests)
clip_valid = 1          # percentage to clip the valid dataset (for tests)
valid_percent = 0.15    # use a percent. of train dataset as validation dataset
test_percent = 0.15     # use a percent. of train dataset as test dataset
start_save = 30         # epoch to start saving
image_height = 300      # height to crop the image
image_width = 300       # width to crop the image
pin_memory = True
load_model = False      # 'true' to load a model and test it, or use it
save_model = False      # 'true' to save model trained after epoches
continue_training = False   # 'true' to load and continue training a model
change_last_fc = False  # to change the last fully connected layer
test_models = False     # 'true' to test the models saved in 'save_results_dir'
last_epoch = 0

# defining the paths to datasets
if run_on_colabs:
    train_image_dir = ['/content/gdrive/SharedDrives/Veterinary Microscope/Dataset/Raabin -WBC - Data Double-labeled Cropped Cells/Train2']
    val_image_dir = ['/content/gdrive/SharedDrives/Veterinary Microscope/Dataset/Raabin -WBC - Data Double-labeled Cropped Cells/Test2']
    csv_file_train =  ['/content/gdrive/SharedDrives/Veterinary Microscope/Dataset/Raabin -WBC - Data Double-labeled Cropped Cells/labels_train.csv']
    csv_file_valid =  ['/content/gdrive/SharedDrives/Veterinary Microscope/Dataset/Raabin -WBC - Data Double-labeled Cropped Cells/labels_test.csv']
else:
    train_image_dir = ['G:/Shared drives/Veterinary Microscope/Dataset/Raabin -WBC - Data Double-labeled Cropped Cells/Train2']
    val_image_dir = ['G:/Shared drives/Veterinary Microscope/Dataset/Raabin -WBC - Data Double-labeled Cropped Cells/Test2']
    csv_file_train =  ['G:/Shared drives/Veterinary Microscope/Dataset/Raabin -WBC - Data Double-labeled Cropped Cells/labels_train.csv']
    csv_file_valid =  ['G:/Shared drives/Veterinary Microscope/Dataset/Raabin -WBC - Data Double-labeled Cropped Cells/labels_test.csv']

# directory to save the results and to test the models:
if run_on_colabs:
    save_results_dir = '/content/gdrive/MyDrive/College/Biophotonics Lab/Research/Programs/Python/Adversarial Attacks/Classification'
    test_models_dir = '/content/gdrive/MyDrive/College/Biophotonics Lab/Research/Programs/Python/Adversarial Attacks/Classification'
else:
    save_results_dir = 'C:/Users/marlo/My Drive/College/Biophotonics Lab/Research/Programs/Python/Adversarial Attacks/Classification'
    test_models_dir = 'C:/Users/marlo/My Drive/College/Biophotonics Lab/Research/Programs/Python/Adversarial Attacks/Classification'

# defining the training function
def train_fn(loader, model, optimizer, loss_fn, scaler, schedule, epoch, last_lr):
    loop = tqdm(loader, desc='Epoch '+str(epoch+1))
    
    for batch_idx, (dictionary, label) in enumerate(loop):
        x, y = dictionary['image0'], label
        # for label to be compared with prediction, as the way it outcomes from the
        # network we need to transform it to 'LongTensor'
        y = y.type(torch.LongTensor)
        x, y = x.to(device=device), y.to(device=device)
        # forward
        with torch.cuda.amp.autocast() if torch.cuda.is_available() else torch.autocast('cpu'):
            pred = model(x)
            loss = loss_fn(pred, y)
        
        # backward
        optimizer.zero_grad()
        if device == 'cuda':
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scale = scaler.get_scale()
            scaler.update()
        else:
            # if device='cpu', we cannot use 'scaler=torch.cuda.amp.GradScaler()'
            loss.backward()
            optimizer.step()

        # updating tqdm loop
        loop.set_postfix(loss=loss.item())
    
    # scheduling learning rate and saving it last value
    if scaler:
        if scale >= scaler.get_scale():
            schedule.step()
            last_lr = schedule.get_last_lr()
    else:
        schedule.step()
        last_lr = schedule.get_last_lr()
    
    return loss.item()*100, last_lr


def main():
    # defining model and casting to device
    model = ResNet50(in_channels=3, num_classes=5).to(device)
    # defining loss function
    loss_fn = nn.CrossEntropyLoss()
    # defining the optimizer (there are commented options below)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    # optimizer = optim.Adam(model.parameters())
    # decay learning rate in a rate of 'gamma' per epoch
    schedule = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    ## To eliminate the scheduling, use 'schedule=None'
    # schedule = None

    # loading dataLoaders
    train_loader, test_loader, valid_loader = get_loaders(
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
    
    if load_model:
        # loading checkpoint, if 'cpu', we need to pass 'map_location'
        os.chdir(root_folder)
        if device == 'cuda':
            load_checkpoint(torch.load('my_checkpoint.pth.tar'), model)
        else:
            load_checkpoint(torch.load('my_checkpoint.pth.tar',
                                       map_location=torch.device('cpu')), model)
        check_accuracy(valid_loader, model, loss_fn, device=device)
    
    if not load_model or continue_training:
        # changing folder to save dictionary
        os.chdir(save_results_dir)
        # if 'continue_training==True', we load the model and continue training
        if continue_training:
            print('- Continue Training...\n')
            start = time.time()
            if device == 'cuda':
                load_checkpoint(torch.load('my_checkpoint.pth.tar'), model,
                                optimizer=optimizer)
            else:
                load_checkpoint(torch.load('my_checkpoint.pth.tar',
                                           map_location=torch.device('cpu')),
                                           model, optimizer=optimizer)
            # reading the csv 'dictionary.csv' as a dictionary
            df = pd.read_csv('dictionary.csv')
            temp = df.to_dict('split')
            temp = temp['data']
            dictionary = {'acc':[], 'loss':[], 'time taken':[]}
            for acc, loss, time_item in temp:
                dictionary['acc'].append(acc)
                dictionary['loss'].append(loss)
                dictionary['time taken'].append(time_item)
            # if change the last fully-connected layer:
            if change_last_fc == True:
                print('yess changes')
                model.fc = nn.Linear(21, 5)
                model.cuda()
        elif not continue_training:
            print('- Start Training...\n')
            start = time.time()
            # opening a 'loss' and 'acc' list, to save the data
            dictionary = {'acc':[], 'loss':[], 'time taken':[]}
            acc_item, loss_item = check_accuracy(valid_loader, model, loss_fn, device=device)
            dictionary['acc'].append(acc_item)
            dictionary['loss'].append(loss_item)
            dictionary['time taken'].append((time.time()-start)/60)
        
        if device == 'cuda':
            scaler = torch.cuda.amp.GradScaler()
        else:
            # with 'cpu' we can't use cuda.amp.GradScaler(), we only use autograd
            scaler = None
        
        # to use 'last_lr' in 'train_fn', we have to define it first
        last_lr = schedule.get_last_lr()
        
        # running epochs for
        for epoch in range(num_epochs):
            loss_item, last_lr = train_fn(train_loader, model, optimizer,
                                          loss_fn, scaler, schedule, epoch,
                                          last_lr)
            
            dictionary['loss'].append(loss_item)
            # save model
            if save_model and epoch >= start_save:
                checkpoint = {
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }
                save_checkpoint(checkpoint, filename='my_checkpoint'+str(epoch+1+last_epoch)+'.pth.tar')
            # check accuracy
            acc_item, temp = check_accuracy(valid_loader, model, loss_fn, device=device)
            stop = time.time()
            dictionary['acc'].append(acc_item)
            dictionary['time taken'].append((stop-start)/60)
            # saving dictionary to csv file
            if save_model:
                df = pd.DataFrame(dictionary, columns = ['acc', 'loss', 'time taken'])
                df.to_csv('dictionary.csv', index = False)
                        
            print('- Time taken:',round((stop-start)/60,3),'min')
            print('- Last Learning rate:', round(last_lr[0],8),'\n')
    
        plt.subplots()
        plt.plot(np.asarray(dictionary['acc'])/100, label ='accuracy')
        plt.plot(np.asarray(dictionary['loss'])/100, label = 'loss')
        plt.legend()
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy and Loss')
        plt.show()


if __name__ == '__main__':
    main()


#%% Defining Testing Function

def testing_models():
        
    model = ResNet50(in_channels=3, num_classes=5).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    schedule = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    # getting the 'valid_loader'
    train_loader, valid_loader = get_loaders(
        train_image_dir=train_image_dir,
        csv_file_train=csv_file_train,
        valid_percent=valid_percent,
        batch_size=batch_size,
        image_height=image_height,
        image_width=image_width,
        num_workers=num_workers,
        pin_memory=pin_memory,
        clip_valid=clip_valid,
        clip_train=clip_train
    )
    
    loss_fn = nn.CrossEntropyLoss()
    
    os.chdir(test_models_dir)
    for file in os.listdir(test_models_dir):
        print('\n\n')
        if 'my_checkpoint' in file:
            # checking accuracy
            if device == 'cuda':
                load_checkpoint(torch.load(file), model)
            else:
                load_checkpoint(torch.load(file,
                                           map_location=torch.device('cpu')),
                                           model)
            print('- Model:', file)
            acc, loss = check_accuracy(valid_loader, model, loss_fn, device=device)
            print('- Acc:',round(acc,3),'; loss:',round(loss,3))

# running testing function if we are in main
if (__name__ == '__main__') and (test_models == True):
    testing_models()
