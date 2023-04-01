# Attacking White-Blood-Cells

## Routine to perform adversarial attacks in a UResNet50 for White-Blood-Cells segmentation

This routine uses Torchattacks library to attack a pre-trained model of 97%
accuracy in segmenting cytoplasm and nuclei in white-blood cells on stained
blood slides. Using the torchattacks library, a diverse range of attacks can
be conducted using this same file, only changing how to call the 'atk' object
with the desired attack method. See 'torchattacks' documentation for more
information.

The 'main.py' file can only run with other three python files, named 'utils.py', 
with the utils functions to be used here, 'dataset.py' to load the dataset
images from a torch DataLoader, and 'model.py' where the actual UResNet50
model can be found. There is also a supplementary 'train.py' file, which can
be used to conduct training of the UResNets (with 18, 34, 50, 101 or 152
layers). All python files are available on GitHub repository (link below)

To attack your model, you need to train it on the Raabin-WBC Dataset using the
'train.py' file, or on other dataset of your choise. This program can also be
the basis to develop automated attacks in other models and other datasets, just
by changing the model file to be loaded and the directories.
