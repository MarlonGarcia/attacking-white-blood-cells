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


@author: Marlon Rodrigues Garcia

"""
# # # Importing Drive
# from google.colab import drive
# drive.mount('/content/gdrive')
# # To import add current folder to path (import py files):
# import sys
root_folder = 'C:/Users/marlo/My Drive/College/Biophotonics Lab/Research/Programs/Python/Adversarial Attacks'
save_results_dir = 'C:/Users/marlo/My Drive/College/Biophotonics Lab/Research/Programs/Python/Adversarial Attacks'
root_model = 'C:/Users/marlo/My Drive/College/Biophotonics Lab/Research/Programs/Python/Adversarial Attacks'
# directory to load models to test, normally equal to 'save_results_dir'
test_models_dir = 'C:/Users/marlo/My Drive/College/Biophotonics Lab/Research/Programs/Python/Adversarial Attacks'
# root_folder = '/content/gdrive/MyDrive/College/Biophotonics Lab/Research/Programs/Python/Adversarial Attacks'
# save_results_dir = '/content/gdrive/MyDrive/College/Biophotonics Lab/Research/Programs/Python/Adversarial Attacks'
# root_model = '/content/gdrive/MyDrive/College/Biophotonics Lab/Research/Programs/Python/Adversarial Attacks'
# sys.path.append(root_folder)

import torch
import os
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torchvision.transforms.functional as tf
import numpy as np
import pandas as pd
from model import UResNet50
from utils import *
import time

# Defining the Hyperparameters
learning_rate = 1e-3    # learning rate
device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 4          # batch size
num_epochs = 85         # number of epochs
num_workers = 3         # number of workers (smaller or = n° processing units)
clip_train = 1.00       # percentage to clip the train dataset (for tests)
clip_valid = 1.00       # percentage to clip the valid dataset (for tests)
valid_percent = 0.15    # use a percent of train dataset as validation dataset
test_percent = 0.15     # a percent from training dataset (but do not excluded)
start_save = 20          # epoch to start saving
image_height = 224      # height to crop the image
image_width = 224       # width to crop the image
pin_memory = True
load_model = True       # 'true' to load a model and test it, or use it
save_model = True       # 'true' to save model trained after epoches
continue_train = True   # 'true' to load and continue training a model
save_images = True      # saving example from predicted and original
change_last_fc = False  # to change the last fully connected layer
test_models = True      # true: test all the models saved in 'save_results_dir'
last_epoch = 81         # when 'continue_train', it is the n° of the last epoch

train_image_dir = ['G:/Shared drives/Veterinary Microscope/Dataset/Raabin-WBC Data - Nucleus_cytoplasm_Ground truths/GrTh/Original/Basophil',
                   'G:/Shared drives/Veterinary Microscope/Dataset/Raabin-WBC Data - Nucleus_cytoplasm_Ground truths/GrTh/Original/Eosinophil',
                   'G:/Shared drives/Veterinary Microscope/Dataset/Raabin-WBC Data - Nucleus_cytoplasm_Ground truths/GrTh/Original/Lymphocyte',
                   'G:/Shared drives/Veterinary Microscope/Dataset/Raabin-WBC Data - Nucleus_cytoplasm_Ground truths/GrTh/Original/Monocyte',
                   'G:/Shared drives/Veterinary Microscope/Dataset/Raabin-WBC Data - Nucleus_cytoplasm_Ground truths/GrTh/Original/Neutrophil']

val_image_dir = None


def train_fn(loader, model, optimizer, loss_fn, scaler, schedule, epoch, last_lr):
    loop = tqdm(loader, desc='Epoch '+str(epoch+1))
    
    for batch_idx, (dictionary) in enumerate(loop):
        image, label = dictionary
        x, y = dictionary[image], dictionary[label]
        x, y = x.to(device=device), y.to(device=device)
        y = y.float()

        # Forward
        with torch.cuda.amp.autocast() if torch.cuda.is_available() else torch.autocast('cpu'):
            pred = model(x)
            y = tf.center_crop(y, pred.shape[2:])
            loss = loss_fn(pred, y)
        
        # Backward
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
        # deliting loss, prediction, x, and y
        loss_item = loss.item()
        del loss, pred, y, x, image, label, dictionary
        #updating tgdm loop
        loop.set_postfix(loss=loss_item)
    # deliting loader and loop
    del loader, loop
    #scheduling learning rate
    if scaler:
        if scale >= scaler.get_scale():
            schedule.step()
            last_lr = schedule.get_last_lr()
    else:
        schedule.step()
        last_lr = schedule.get_last_lr()
    
    return loss_item, last_lr


def main():
    
    model = UResNet50(in_channels=3, num_classes=3).to(device)
    # Use the optimizer BCEWithLogitsLoss for binery classification, without
    # using logitic functio inside the model (it executs the logistic for you).
    # If we want to multclass segmentation, we change the 'out_channels' to the
    # number of classes, and the 'loss_fn' to cross entropy loss (ading a lo-
    # gistic function inside your model)
    # loss_fn = nn.BCEWithLogitsLoss()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())#, lr=learning_rate)
    # optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    schedule = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)
    # schedule = None
    
    train_loader, test_loader, valid_loader = get_loaders(
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
    
    if load_model:
        # Loading checkpoint, if 'cpu', we need to pass 'map_location'
        os.chdir(root_folder)
        if device == 'cuda':
            load_checkpoint(torch.load('my_checkpoint.pth.tar'), model)
        else:
            load_checkpoint(torch.load('my_checkpoint.pth.tar',
                                       map_location=torch.device('cpu')), model)
        check_accuracy(valid_loader, model, loss_fn, device=device)
    
    if not load_model or continue_train:
        # Changing folder to save dictionary
        os.chdir(save_results_dir)
        # If 'continue_train = True', we load the model and continue training
        if continue_train:
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
            dictionary = {'acc-valid':[], 'acc-test':[], 'loss':[], 'dice score-valid':[], 'dice score-test':[], 'time taken':[]}
            for acc_valid, acc_test, loss, dice_score_valid, dice_score_test, time_item in temp:
                dictionary['acc-valid'].append(acc_valid)
                dictionary['acc-test'].append(acc_test)
                dictionary['loss'].append(loss)
                dictionary['dice score-valid'].append(dice_score_valid)
                dictionary['dice score-test'].append(dice_score_test)
                dictionary['time taken'].append(time_item)
            # Add a last time to continue conting from here
            last_time = time_item
            # if change the last fully-connected layer:
            if change_last_fc == True:
                print('- Change last fully-connected layer')
                model.fc = nn.Linear(21, 5)
                model.cuda()
        elif not continue_train:
            print('- Start Training...\n')
            start = time.time()
            # Opening a 'loss' and 'acc' list, to save the data
            dictionary = {'acc-valid':[], 'acc-test':[], 'loss':[], 'dice score-valid':[], 'dice score-test':[], 'time taken':[]}
            acc_item_valid, loss_item, dice_score_valid = check_accuracy(valid_loader, model, loss_fn, device=device, title='Validating')
            acc_item_test, _, dice_score_test = check_accuracy(test_loader, model, loss_fn, device=device, title='Testing')
            print('\n')
            dictionary['acc-valid'].append(acc_item_valid)
            dictionary['acc-test'].append(acc_item_test)
            dictionary['loss'].append(loss_item)
            dictionary['dice score-valid'].append(dice_score_valid)
            dictionary['dice score-test'].append(dice_score_test)
            # Adding a last_time here, in the case we do not continue a train
            last_time = (time.time()-start)/60
            dictionary['time taken'].append(last_time)
        
        if device == 'cuda':
            scaler = torch.cuda.amp.GradScaler()
        else:
            # with 'cpu' we can't use cuda.amp.GradScaler(), we only use autograd
            scaler = None
        
        # to use 'last_lr' in 'train_fn', we have to define it first
        last_lr = schedule.get_last_lr()
        
        # begining image printing
        fig, ax = plt.subplots()
        
        # Criating a new start time (we have to sum this to 'last_time')
        start = time.time()
        
        # running epochs for
        for epoch in range(last_epoch, num_epochs):
            
            loss_item, last_lr = train_fn(train_loader, model, optimizer,
                                          loss_fn, scaler, schedule, epoch,
                                          last_lr)
            
            dictionary['loss'].append(loss_item)
            # save model
            if save_model and epoch >= start_save -1:
                checkpoint = {
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }
                save_checkpoint(checkpoint, filename='my_checkpoint'+str(epoch+1)+'.pth.tar')
            # check accuracy
            print('\nValidating:')
            acc_item_valid, _, dice_score_valid = check_accuracy(valid_loader, model, loss_fn, device=device)
            print('Testing:')
            acc_item_test, _, dice_score_test = check_accuracy(test_loader, model, loss_fn, device=device)
            stop = time.time()
            dictionary['acc-valid'].append(acc_item_valid)
            dictionary['acc-test'].append(acc_item_test)
            dictionary['dice score-valid'].append(dice_score_valid)
            dictionary['dice score-test'].append(dice_score_test)
            dictionary['time taken'].append((stop-start)/60+last_time)
            # print some examples to folder
            if save_images:
                save_predictions_as_imgs(
                    valid_loader, model, folder=os.path.join(root_folder,'saved_images'),
                    device=device
                )
            # saving dictionary to csv file
            if save_model:
                df = pd.DataFrame(dictionary, columns = ['acc-valid', 'acc-test',
                                                         'loss', 'dice score-valid',
                                                         'dice score-test', 'time taken'])
                df.to_csv('dictionary.csv', index = False)
                        
            print('- Time taken:',round((stop-start)/60+last_time,3),'min')
            print('- Last Learning rate:', round(last_lr[0],8),'\n\n')
            # Deliting variables
            del dice_score_test, dice_score_valid, acc_item_test, acc_item_valid, loss_item, stop
            try: del checkpoint
            except: pass
            
            # Continue image printing
            if epoch == last_epoch:
                ax.plot(np.asarray(dictionary['acc-valid']), 'C1', label ='accuracy-validation')
                ax.plot(np.asarray(dictionary['acc-test']), 'C2', label ='accuracy-test')
                ax.plot(np.asarray(dictionary['dice score-valid']), 'C4', label = 'dice score-validation')
                ax.plot(np.asarray(dictionary['dice score-test']), 'C5', label = 'dice score-test')
                ax.plot(np.asarray(dictionary['loss']), 'C3', label = 'loss')
                plt.legend()
                ax.set_xlabel('Epochs')
                ax.set_ylabel('Accuracy, Loss, and Dice score')
                plt.pause(0.5)
            else:
                ax.plot(np.asarray(dictionary['acc-valid']), 'C1')
                ax.plot(np.asarray(dictionary['acc-test']), 'C2')
                ax.plot(np.asarray(dictionary['dice score-valid']), 'C4')
                ax.plot(np.asarray(dictionary['dice score-test']), 'C5')
                ax.plot(np.asarray(dictionary['loss']), 'C3')
            plt.show()
            plt.pause(0.5)


if __name__ == '__main__':
    main()



def testing_models():
        
    model = UResNet50(in_channels=3, num_classes=3).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    schedule = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)
    # getting the 'valid_loader'
    train_loader, valid_loader = get_loaders(
        train_image_dir=train_image_dir,
        csv_file_train=csv_file_train,
        valid_percent=valid_percent,
        test_percent=test_percent,
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

if (__name__ == '__main__') and (test_models == True):
    testing_models()
    