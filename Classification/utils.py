'''
This file is used together with the 'train.py' file to help in the training and
testing process with util functions.
'''
import torch
from dataset import RaabinDataset
from torch.utils.data import DataLoader, random_split
import torchvision.transforms.functional as tf
from torchvision.transforms import Compose
from tqdm import tqdm
import random


# The next functions are functional transforms, used to apply functions in a way
# controled by the user. So we can apply, for example, in the data image and in
# the label image (so it is called deterministic, because we can determine the
# same transformation to be applied in more then one image). This is the unique
# way to apply the same transformation to more then one different image in torch

class ToTensor(object):
    '''Function to transform a ndarray in a tensor
    
    n: int (input)
        number of non-mask images to convert to tensor (the rest will be
        converted without scaling to [0.0,1.0])'''
    def __init__(self, n):
        self.n = n
        
    def __call__(self, images):
        for i, image in enumerate(images):
            if i < self.n:
                images[image] = tf.to_tensor(images[image])
            else:
                images[image] = torch.from_numpy(images[image])
                images[image] = torch.permute(images[image], (2,0,1))
                
        return images


class Rotate(object):
    '''Function to rotate an image, the input is a dictionary
    
    images: 'dictionary' (input)
        dictionary with images;
    limit: 'list'
        a list 'int' with smaller and larger angles to rotate (e.g. [0, 90]);
    p: 'float'
        probability to rotate;
    
    dictionary: 'dictionary' (output)
        dictionary with cropped images with keys 'image0', 'image1', etc.
    '''
    def __init__(self,**kwargs):
        # 'limit' is a 'list' that defines the lower and upper angular limits
        limit = kwargs.get('limit')
        if not limit: limit = [0, 360]
        self.limit = limit
        # 'p' is 'float' the probability to happen a rotate
        p = kwargs.get('p')
        if not p: p = 0.5
        self.p = p
    
    def __call__(self, images):
        if random.random() > 1-self.p:
            angle = random.randint(self.limit[0], self.limit[1])
            for i, image in enumerate(images):
                images[image] = tf.rotate(images[image], angle)
        
        return images


class CenterCrop(object):
    '''Function to center crop one or multiple images
    
    size: 'list' (input)
        input list with size (e.g. '[400,200]');
    images: 'dictionary' (input) (output)
        dictionary with images.
    '''
    def __init__(self, size):
        self.size = size
        
        
    def __call__(self, images):
        for image in images:
            images[image] = tf.center_crop(images[image], self.size)
        
        return images


class Resize(object):
    '''Function to resize one or multiple images
    
    size: 'list' (input)
        input list with size (e.g. '[400,200]');
    images: 'dictionary' (input) (output)
        dictionary with images.
    '''
    def __init__(self, size):
        self.size = size
    
    def __call__(self, images):
        for image in images:
            images[image] = tf.resize(images[image], self.size)
    
        return images


class FlipHorizontal(object):
    '''Horizontally flip images randomly
    
    p: 'float' (input)
        probability to flip (from 0.0 to 1.0).
    '''
    def __init__(self, p):
        self.p = p
    
    def __call__(self, images):
        if random.random() > 1-self.p:
            for image in images:
                images[image] = tf.hflip(images[image])
        
        return images


class FlipVertical(object):
    '''Vertically flip images randomly
    
    p: 'float' (input)
        probability to flip (from 0.0 to 1.0).
    '''
    def __init__(self, p):
        self.p = p
    
    def __call__(self, images):
        if random.random() > 1-self.p:
            for image in images:
                images[image] = tf.vflip(images[image])
        
        return images


class Normalize(object):
    '''Normalizing 'n' images of a given set of images
    
    n: int (input)
        number of images to normalize;
    mean: list (input)
        mean to normalize;
    std: list (input)
        stadard deviation to normalize.
    '''
    def __init__(self, n=1, mean=0.5, std=0.5):
        self.n = n
        self.mean = mean
        self.std = std
    
    def __call__(self, images):
        for i, image in enumerate(images):
            if i < self.n:
                images[image] = tf.normalize(images[image], self.mean, self.std)
        
        return images


class Affine(object):
    '''Affining images
    
    size: list (input)
        maximum higher and width to translate image (normally the image size);
    scale: float (input)
        scale to perform affine (between 0 and 1.0);
    p: float (input)
        probability to thange.'''
    def __init__(self, size=[0,0], scale=0.5, p=0.5):
        self.size = size
        self.scale = scale
        self.p = p
    
    def __call__(self, images):
        if random.random() > 1-self.p:
            angle = random.random()*self.scale*360
            shear = random.random()*self.scale*360
            translate = [i*random.random()*self.scale for i in self.size]
            for image in images:
                images[image] = tf.affine(images[image], angle=angle,
                                         translate=translate, scale=1-self.scale,
                                         shear=shear)
        return images


#%% Util Functions to be Used During Training or Testing

# saving checkpoints
def save_checkpoint(state, filename='my_checkpoint.pth.tar'):
    print('- Saving Checkpoint...')
    torch.save(state, filename)

# loading checkpoints
def load_checkpoint(checkpoint, model, optimizer=None):
    print('- Loading Checkpoint...')
    model.load_state_dict(checkpoint['state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer'])

# getting loaders given directories and other informations
def get_loaders(train_image_dir,
                csv_file_train,
                valid_percent,
                test_percent,
                batch_size,
                image_height,
                image_width,
                num_workers=1,
                pin_memory=True,
                val_image_dir=None,
                csv_file_valid=None,
                clip_valid=1.0,
                clip_train=1.0):
    
    # first, defining transformations to be applied in the train images to be loaded
    transform_train_0 = Compose([ToTensor(n=1),
                                 Resize(size=[image_height, image_width]),
                                 FlipVertical(p=0.5),
                                 FlipHorizontal(p=0.5),
                                 # RaabinDataset for classifi (train dataset):
                                 Normalize(n=1, mean=[0.6547, 0.5921, 0.5816],
                                           std=[0.10992, 0.12602, 0.04328])]
                                )
    # defining the same, but for validation and testing images (can be different)
    transform_valid_0 = Compose([ToTensor(n=1),
                                 Resize(size=[image_height, image_width]),
                                 FlipVertical(p=0.5),
                                 FlipHorizontal(p=0.5),
                                 # RaabinDataset for classifi (valid dataset):
                                 Normalize(n=1, mean=[0.4969, 0.29839, 0.49723],
                                           std=[0.10984, 0.14822, 0.043955])]
                                )
    
    # second, defining the number of transformations per directory in 'train_image_dir'
    transformations_per_dataset = [1, 1, 1, 1, 1, 1, 1]
    
    # third, reading the dataset in a as a 'torch.utils.data.Dataset' instance.
    # it is only for images in 'train_image_dir[0]', next we accounts for the rest
    train_dataset = RaabinDataset(image_dir=train_image_dir[0],
                                  csv_file=csv_file_train[0],
                                  transform=transform_train_0)
    
    # concatenate the other directories in 'train_image_dir[:]' in a larger
    # 'torhc.utils.data.Dataset'. after we concatenate more to augment the data
    for n in range(1, len(train_image_dir)):
        dataset_train_temp = RaabinDataset(image_dir=train_image_dir[n],
                                           csv_file=csv_file_train[n],
                                           transform=transform_train_0)
        # to use 'train_dataset' here in right, we have to define it before
        train_dataset = torch.utils.data.ConcatDataset([train_dataset,
                                                        dataset_train_temp])
    
    # using part of the training data as test datset
    test_dataset_size = int(test_percent*len(train_dataset))
    rest_size = int((1-test_percent)*len(train_dataset))
    if test_dataset_size+rest_size != len(train_dataset):
        rest_size += 1
    (test_dataset, _) = random_split(train_dataset, [test_dataset_size, rest_size],
                                     generator=(torch.Generator().manual_seed(40)))
    
    # defining the validation dataset, using part of 'train_dataset', or using
    # a specific dataset for validation, if 'val_image_dir' is not 'None'
    if not val_image_dir:
        valid_dataset_size = int(valid_percent*len(train_dataset))
        train_dataset_size = int((1-valid_percent)*len(train_dataset))
        # adding one to train_dataset_size if 'int' operation removed it
        if valid_dataset_size+train_dataset_size != len(train_dataset):
            train_dataset_size += 1
        (train_dataset, valid_dataset) = random_split(train_dataset,
                                         [train_dataset_size, valid_dataset_size],
                                         generator=torch.Generator().manual_seed(20))
    else:
        valid_dataset = RaabinDataset(image_dir=val_image_dir[0],
                                      csv_file=csv_file_valid[0],
                                      transform=transform_valid_0)
        for n in range(1, len(val_image_dir)):
            dataset_val_temp = RaabinDataset(image_dir=val_image_dir[n],
                                             csv_file=csv_file_valid[n],
                                             transform=transform_valid_0)
            valid_dataset = torch.utils.data.ConcatDataset([valid_dataset,
                                                            dataset_val_temp])
    
    # concatenating the augmented data, in case 'transf..._per_dataset' > 1
    for n in range(0,len(train_image_dir)):
        for m in range(1, transformations_per_dataset[n]):
            # first we specify the transformation (depending on the 'm' value)
            if m < 2:
                transformation = Compose([ToTensor(n=1),
                                          Rotate(limit=[(m-1)*72,m*72], p=1.0),
                                          Resize(size=[image_height, image_width]),
                                          FlipVertical(p=0.5),
                                          FlipHorizontal(p=0.5),
                                          # RaabinDataset for classifi (train dataset):
                                          Normalize(n=1,mean=[0.6547, 0.5921, 0.5816],
                                                    std=[0.10992, 0.12602, 0.04328])]
                                          )
            else:
                transformation = Compose([ToTensor(n=1),
                                          Affine(size=[0.5*image_height, 0.5*image_width],
                                                 scale=0.01*(m-1), p=0.5),
                                          Rotate(limit=[(m-1)*72,m*72], p=1.0),
                                          Resize(size=[image_height, image_width]),
                                          FlipVertical(p=0.5),
                                          FlipHorizontal(p=0.5),
                                          # RaabinDataset for classifi (train dataset):
                                          Normalize(n=1,mean=[0.6547, 0.5921, 0.5816],
                                                    std=[0.10992, 0.12602, 0.04328])]
                                          )
            # then we apply this transformation to read the dataset as 'torch.utils.data.Dataset'
            dataset_train_temp = RaabinDataset(image_dir=train_image_dir[n],
                                               csv_file=csv_file_train[n],
                                               transform=transformation)
            train_dataset = torch.utils.data.ConcatDataset([train_dataset, dataset_train_temp])
    
    # splitting the dataset, to deminish if 'clip_valid'<1 for fast testing
    if clip_train < 1:
        print('- Splitting Training Dataset ',clip_train*100,'%')
        train_mini = int(clip_train*len(train_dataset))
        temp_mini = int((1-clip_train)*len(train_dataset))
        if train_mini+temp_mini != len(train_dataset):
            temp_mini += 1
        (train_dataset, _) = random_split(train_dataset,[train_mini, temp_mini],
                                             generator=torch.Generator().manual_seed(40))
    if clip_valid < 1:
        print('- Splitting Validation Dataset ',clip_valid*100,'%')
        valid_mini = int(clip_valid*len(valid_dataset))
        temp_mini = int((1-clip_valid)*len(valid_dataset))
        if valid_mini+temp_mini != len(valid_dataset):
            temp_mini += 1
        (valid_dataset, _) = random_split(valid_dataset,[valid_mini, temp_mini],
                                             generator=torch.Generator().manual_seed(30))
        (test_dataset, _) = random_split(test_dataset, [valid_mini, temp_mini],
                                         generator=torch.Generator().manual_seed(50))
    
    # obtaining dataloader from dataset
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              num_workers=num_workers,
                              pin_memory=pin_memory, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                              num_workers=num_workers,
                              pin_memory=pin_memory)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size,
                              num_workers=num_workers,
                              pin_memory=pin_memory)
    
    return train_loader, test_loader, valid_loader

# defining function to check the accuracy
def check_accuracy(loader, model, loss_fn, device='cuda' if torch.cuda.is_available() else 'cpu'):
    num_correct = 0
    model.eval()
    
    loop = tqdm(loader, desc='Check acc')
    
    with torch.no_grad():
        for dictionary, label in loop:
            x, y = dictionary['image0'], label
            x = x.to(device=device)
            y = y.type(torch.LongTensor)
            y = y.to(device=device)
            pred = model(x)
            loss = loss_fn(pred, y)
            num_correct += (pred.argmax(1)==y).type(torch.float).sum().item()
            loop.set_postfix(acc=str(round(100*num_correct/len(loader.dataset),4)))
    
    print(f'\nGot an accuracy of {round(100*num_correct/len(loader.dataset),4)}')
    # model is training when entering this function.
    model.train()
    
    return 100*num_correct/len(loader.dataset), loss.item()*100
