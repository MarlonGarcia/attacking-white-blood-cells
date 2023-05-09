# -*- coding: utf-8 -*-
"""
Program to Train, Save and Continue Training of ANN Models

This program is capable of training and saving models, also continue a training
from last epoch (use hyperparameter 'last_epoch' and 'continue_training=True',
and 'load_model=True'), and to save resulting images (when used to images, bur
it is general porpouse), using 'save_images=True'.
                                                  
This program suports the optional use of a validation dataset, which directory
has to be passed through 'val_image_dir'. If this variable is 'None', then test
and validation datasets is splitted from train dataset using 'valid_percent'
and 'test_percent'. The test dataset is always clipped from the training data-
set.

Transfer Learning: In the case fully-connected layers are used, and you are
using a pre-traiend model, and want to add an extra fully-connected layer for
transfer learning, you can perform it by changing the 'change_last_fc' variable
to 'True'.

When just testing model(s), we just change 'laod_model' to 'True', and run the
program with the variable 'test_models_dir' assigned with the directories where
the models to be tested are.

When using 'continue_training = True' in the hyperparameters, you will continue
the training of a model. In order to that, the last epoch has to be stated in
'last_epoch' variable, and the name of the pre-trained model has to be exactly
'my_checkpoint.pth.tar'. The variable 'laod_model' does not need to be 'True'.

For a normal traning from a non-trained model (from first epoch), we use the
default values of 'continue_training = False' and 'change_last_fc = False'. In
this case if 'load_model = True', the models will be tested afther last epoch.

Find more on the GitHub Repository:
https://github.com/MarlonGarcia/attacking-white-blood-cells

@author: Marlon Rodrigues Garcia
@instit: University of São Paulo
"""

### Program Header

import torch
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
    root_folder = '/content/gdrive/MyDrive/College/Biophotonics Lab/Research/Programs/Python/Adversarial Attacks/attacking-white-blood-cells/attacking-white-blood-cells/Classification'
    sys.path.append(root_folder)
else:
    root_folder = 'C:/Users/marlo/My Drive/College/Biophotonics Lab/Research/Programs/Python/Adversarial Attacks/attacking-white-blood-cells/attacking-white-blood-cells/Classification'
import os
os.chdir(root_folder)
from utils import *
from model import ResNet18


#%% Defining Parameters and Path

# defining hyperparameters
learning_rate = 1e-3    # learning rate
device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 8          # batch size
num_epochs = 60         # number of epochs
num_workers = 3         # number of workers
clip_train = 0.5        # percentage to clip the train dataset (for tests)
clip_valid = 0.5        # percentage to clip the valid dataset (for tests)
valid_percent = 0.15    # use a percent. of train dataset as validation dataset
test_percent = 0.15     # use a percent. of train dataset as test dataset
start_save = 2          # epoch to start saving
image_height = 300      # height to crop the image
image_width = 300       # width to crop the image
pin_memory = True
load_model = False      # 'true' to load a model and test it, or use it
save_model = True       # 'true' to save model trained after epoches
continue_training = False # 'true' to load and continue training a model
change_last_fc = False  # to change the last fully connected layer
test_models = False     # 'true' to test the models saved in 'save_results_dir'
last_epoch = 0          # when 'continue_training', it has to be the last epoch

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
    save_results_dir = '/content/gdrive/MyDrive/College/Biophotonics Lab/Research/Programs/Python/Adversarial Attacks/attacking-white-blood-cells/attacking-white-blood-cells/Classification'
    test_models_dir = '/content/gdrive/MyDrive/College/Biophotonics Lab/Research/Programs/Python/Adversarial Attacks/attacking-white-blood-cells/attacking-white-blood-cells/Classification'
else:
    save_results_dir = 'C:/Users/marlo/My Drive/College/Biophotonics Lab/Research/Programs/Python/Adversarial Attacks/attacking-white-blood-cells/attacking-white-blood-cells/Classification'
    test_models_dir = 'C:/Users/marlo/My Drive/College/Biophotonics Lab/Research/Programs/Python/Adversarial Attacks/attacking-white-blood-cells/attacking-white-blood-cells/Classification'

#%% Training Function

# defining the training function
def train_fn(loader, model, optimizer, loss_fn, scaler, schedule, epoch, last_lr):
    loop = tqdm(loader, desc='Epoch '+str(epoch+1))
    
    for batch_idx, (dictionary, label) in enumerate(loop):
        x, y = dictionary['image0'], label
        # transform label to 'LongTensor' for comparison with prediction
        y = y.type(torch.LongTensor)
        x, y = x.to(device=device), y.to(device=device)
        # forward
        with torch.cuda.amp.autocast() if torch.cuda.is_available() else torch.autocast('cpu'):
            pred = model(x)
            # calculating loss
            loss = loss_fn(pred, y)
        
        # backward
        optimizer.zero_grad()
        if device == 'cuda':
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scale = scaler.get_scale()
            scaler.update()
        # if device='cpu', we cannot use 'scaler=torch.cuda.amp.GradScaler()':
        else:
            loss.backward()
            optimizer.step()
        # freeing space by deleting variables
        loss_item = loss.item()
        del loss, pred, y, x, label, dictionary
        # updating tqdm loop
        loop.set_postfix(loss=loss_item)
    # deliting loader and loop
    del loader, loop
    # scheduling the learning rate and saving its last value
    if scaler:
        if scale >= scaler.get_scale():
            schedule.step()
            last_lr = schedule.get_last_lr()
    else:
        schedule.step()
        last_lr = schedule.get_last_lr()
    
    return loss_item, last_lr


#%% Defining The main() Function

def main():
    # defining the model and casting to device
    model = ResNet18(in_channels=3, num_classes=5).to(device)
    # defining loss function
    loss_fn = nn.CrossEntropyLoss()
    # defining the optimizer (there are options commented below)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    # optimizer = optim.Adam(model.parameters())
    # decay learning rate by a rate of 'gamma' per epoch
    schedule = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    # if schedule is not used, please refer it as 'None'
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
    
    # if this program is just to load and test a model, next it loads a model
    if load_model:
        # loading checkpoint
        os.chdir(root_folder)
        if device == 'cuda':
            load_checkpoint(torch.load('my_checkpoint.pth.tar'), model)
        # if 'cpu', we need to pass 'map_location'
        else:
            load_checkpoint(torch.load('my_checkpoint.pth.tar',
                                       map_location=torch.device('cpu')), model)
        check_accuracy(valid_loader, model, loss_fn, device=device)
    
    if not load_model or continue_training:
        # changing folder to save dictionary
        os.chdir(save_results_dir)
        # if 'continue_training==True', we load the model and continue training
        if continue_training:
            print('\n- Continue Training...\n')
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
            dictionary = {'acc-valid':[], 'acc-test':[], 'loss':[], 'time taken':[]}
            for acc_valid, acc_test, loss, time_item in temp:
                dictionary['acc-valid'].append(acc_valid)
                dictionary['acc-test'].append(acc_test)
                dictionary['loss'].append(loss)
                dictionary['time taken'].append(time_item)
            # adding a last time to continue conting from here
            last_time = time_item
            # if change the last fully-connected layer:
            if change_last_fc == True:
                print('\n- Changing last fully-connected layer')
                model.fc = nn.Linear(21, 5)
                model.cuda()
        # if it is the first epoch
        elif not continue_training:
            print('\n- Start Training...\n')
            start = time.time()
            # opening a 'loss' and 'acc' list, to save the data
            dictionary = {'acc-valid':[], 'acc-test':[], 'loss':[], 'time taken':[]}
            acc_item_valid, loss_item = check_accuracy(valid_loader, model, loss_fn, device=device, title='Validating')
            acc_item_test, _ = check_accuracy(test_loader, model, loss_fn, device=device, title='Testing')
            dictionary['acc-valid'].append(acc_item_valid)
            dictionary['acc-test'].append(acc_item_test)
            dictionary['loss'].append(loss_item)
            # we added last_time here to sum it to the 'time taken' in the
            # dictionary. it is done because if training is continued, we can
            # sum the actual 'last_time' taken in previous training.
            last_time = (time.time()-start)/60
            dictionary['time taken'].append(last_time)
        
        # with 'cpu' we can't use 'torch.cuda.amp.GradScaler()'
        if device == 'cuda':
            scaler = torch.cuda.amp.GradScaler()
        else:
            scaler = None
        # to use 'last_lr' in 'train_fn', we have to define it first
        last_lr = schedule.get_last_lr()
        # begining image printing
        fig, ax = plt.subplots()
        # Criating a new start time (we have to sum this to 'last_time')
        start = time.time()
        
        # running epochs
        for epoch in range(last_epoch, num_epochs):
            # calling training function
            loss_item, last_lr = train_fn(train_loader, model, optimizer,
                                          loss_fn, scaler, schedule, epoch,
                                          last_lr)
            # appending resulted loss from training
            dictionary['loss'].append(loss_item)
            # saving model
            if save_model and epoch >= start_save -1:
                # changing folder to save dictionary
                os.chdir(save_results_dir)
                checkpoint = {
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }
                save_checkpoint(checkpoint, filename='my_checkpoint'+str(epoch+1)+'.pth.tar')
            # check accuracy
            print('\nValidating:')
            acc_item_valid, _ = check_accuracy(valid_loader, model, loss_fn, device=device)
            print('Testing:')
            acc_item_test, _ = check_accuracy(test_loader, model, loss_fn, device=device)
            stop = time.time()
            dictionary['acc-valid'].append(acc_item_valid)
            dictionary['acc-test'].append(acc_item_test)
            dictionary['time taken'].append((stop-start)/60+last_time)
            # saving dictionary to a csv file
            if save_model:
                # changing folder to save dictionary
                os.chdir(save_results_dir)
                df = pd.DataFrame(dictionary, columns = ['acc-valid', 'acc-test',
                                                         'loss', 'time taken'])
                df.to_csv('dictionary.csv', index = False)
                        
            print('\n- Time taken:',round((stop-start)/60+last_time,3),'min')
            print('\n- Last Learning rate:', round(last_lr[0],8),'\n\n')
            # deleting variables for freeing space
            del acc_item_valid, acc_item_test, loss_item, stop
            try: del checkpoint
            except: pass
            
            # continue image printing
            if epoch == last_epoch:
                ax.plot(np.asarray(dictionary['acc-valid']), 'C1', label ='accuracy-validation')
                ax.plot(np.asarray(dictionary['acc-test']), 'C2', label ='accuracy-test')
                ax.plot(np.asarray(dictionary['loss']), 'C3', label = 'loss')
                plt.legend()
                ax.set_xlabel('Epochs')
                ax.set_ylabel('Accuracy, Loss, and Dice score')
                plt.pause(0.5)
            else:
                ax.plot(np.asarray(dictionary['acc-valid']), 'C1')
                ax.plot(np.asarray(dictionary['acc-test']), 'C2')
                ax.plot(np.asarray(dictionary['loss']), 'C3')
            plt.show()
            plt.pause(0.5)


if __name__ == '__main__':
    main()


#%% Defining Testing Function

def testing_models():
        
    model = ResNet18(in_channels=3, num_classes=5).to(device)
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
