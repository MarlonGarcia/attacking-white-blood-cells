# Attacking White-Blood-Cells

## Routine to perform adversarial attacks in a UResNet50 for White-Blood-Cells segmentation

This routine uses Torchattacks library to attack a pre-trained model of 97% accuracy in segmenting cytoplasm and nuclei in white-blood cells on stained blood slides. Using the torchattacks library, a diverse range of attacks can be conducted using this same file, only changing how to call the 'atk' object with the desired attack method. See 'torchattacks' documentation for more information.

The 'main.py' file can only run with other three python files, named 'utils.py', with the utils functions to be used here, 'dataset.py' to load the dataset images from a torch DataLoader, and 'model.py' where the actual UResNet50 model can be found. There is also a supplementary 'train.py' file, which can be used to conduct training of the UResNets (with 18, 34, 50, 101 or 152 layers).

To attack your model, you need to train it on the Raabin-WBC Dataset using the 'train.py' file, or on other dataset of your choise. This program can also be the basis to develop automated attacks in other models and other datasets, just by changing the model file to be loaded and the directories.


## Results from Classification Attacks

### Classification Attacks

Attacks performed in a ResNet50 classification model trained to detect types of white blood cells showed accuracy dropping for epsilons greater than 0.1, as we can see by Fig. 1. Examples of original and perturbed images can be seen in Fig. 2.

<!-- ![Alt text](Classification/Images/AccClassif.png) -->
<img src="Classification/Images/AccClassif.png" width="500" height="371">
Figure 1 - Accuracies obtained for the classification model ResNet50 under attack, for the validation and the test data sets, considering different values of ε.

![Alt text](Classification/Images/images.png)
Figure 2 - Example of non-perturbed and perturbed images for different values of ε.


### Segmentation Attacks

Attacks in a UResNet50 (UNet with ResNet encoder and decoder part) segmentation model resulted in less accuracy and dice-score dropping, showing that this model is more robust for this type of attacks (Fig. 3). Fig. 4 shows original and perturbed images exemples.

<!-- ![Alt text](Segmentation/Images/AccSegme.png) -->
<img src="Segmentation/Images/AccSegme.png" width="500" height="377">
Figure 3 - Accuracy and Dice-Score curve, obtained for the segmentation task, for a UResNet50 model attacked in both the validation and the test data sets, for different values of ε.

![Alt text](Segmentation/Images/images.png)
Figure 4 - Examples of perturbed images, and the resulting labels, for different values of ε.