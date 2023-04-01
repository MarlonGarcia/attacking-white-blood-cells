import torch
from dataset import RaabinDataset
from torch.utils.data import DataLoader, random_split
import torchvision.transforms.functional as tf
from torchvision.transforms import Compose
from torchvision.utils import save_image
from tqdm import tqdm
import random
from torchmetrics import Dice


# The next functions are to define classes of functional transforms, used to 
# apply a deterministic transformation in more than one entry, e.g. a random
# rotation, that can be equaly applied for an image and its label (it neets to
# be deterministic fot the label, the same of the image).

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

## Util functions
def save_checkpoint(state, filename='my_checkpoint.pth.tar'):
    print('- Saving Checkpoint...')
    torch.save(state, filename)

def load_checkpoint(checkpoint, model, optimizer=None):
    print('- Loading Checkpoint...')
    model.load_state_dict(checkpoint['state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer'])

def get_loaders(train_image_dir,
                valid_percent,
                test_percent,
                batch_size,
                image_height,
                image_width,
                num_workers=1,
                pin_memory=True,
                val_image_dir=None,
                clip_valid=1.0,
                clip_train=1.0):
    
    transform_train_0 = Compose([ToTensor(n=1),
                                 Resize(size=[image_height, image_width]),
                                 FlipVertical(p=0.5),
                                 FlipHorizontal(p=0.5),
                                 # Mean and std, obtained from the dataset
                                  Normalize(n=1, mean=[0.52096, 0.51698, 0.545980],
                                                  std=[0.10380, 0.11190, 0.118877])]
                                )
    
    transform_valid_0 = Compose([ToTensor(n=1),
                                 Resize(size=[image_height, image_width]),
                                 FlipVertical(p=0.5),
                                 FlipHorizontal(p=0.5),
                                 # Mean and std, obtained from the dataset
                                  Normalize(n=1, mean=[0.52096, 0.51698, 0.545980],
                                                  std=[0.10380, 0.11190, 0.118877])]
                                )
    
    # Number of transformations per dataset
    transformations_per_dataset = [1, 1, 1, 1, 1, 1, 1]
    
    # Firsta part of whole dataset, only with images in 'train_image_dir[0]'
    train_dataset = RaabinDataset(image_dir=train_image_dir[0],
                                  transform=transform_train_0)
    
    # Concatenate in 'train_image_dir', that are different datasets to be added
    # the other for in 'transf.._per_dataset' will be to data augmentation (it
    # is not different datasets, is to really augment the data)
    for n in range(1, len(train_image_dir)):
        dataset_train_temp = RaabinDataset(image_dir=train_image_dir[n],
                                           transform=transform_train_0)
        # to use 'train_dataset' here in right, we have to define it before
        train_dataset = torch.utils.data.ConcatDataset([train_dataset,
                                                        dataset_train_temp])
    
    # Using part of the training data as test datset
    test_dataset_size = int(test_percent*len(train_dataset))
    rest_size = int((1-test_percent)*len(train_dataset))
    if test_dataset_size+rest_size != len(train_dataset):
        rest_size += 1
    (test_dataset, _) = random_split(train_dataset, [test_dataset_size, rest_size],
                                     generator=(torch.Generator().manual_seed(40)))
    
    
    # Defining the validation dataset, using part of 'train_dataset', or using
    # a specific dataset for validation, if 'val_image_dir' is not 'None'
    if not val_image_dir:
        valid_dataset_size = int(valid_percent*len(train_dataset))
        train_dataset_size = int((1-valid_percent)*len(train_dataset))
        # Adding one to train_dataset_size if 'int' operation removed it
        if valid_dataset_size+train_dataset_size != len(train_dataset):
            train_dataset_size += 1
        (train_dataset, valid_dataset) = random_split(train_dataset,
                                         [train_dataset_size, valid_dataset_size],
                                         generator=torch.Generator().manual_seed(20))
    else:
        valid_dataset = RaabinDataset(image_dir=val_image_dir[0],
                                      transform=transform_valid_0)
        for n in range(1, len(val_image_dir)):
            dataset_val_temp = RaabinDataset(image_dir=val_image_dir[n],
                                             transform=transform_valid_0)
            valid_dataset = torch.utils.data.ConcatDataset([valid_dataset,
                                                            dataset_val_temp])
    
    # Following ther rest of augmented data
    for n in range(0,len(train_image_dir)):
        for m in range(1, transformations_per_dataset[n]):
            if m < 2:
                transformation = Compose([ToTensor(n=1),
                                          Rotate(limit=[(m-1)*72,m*72], p=1.0),
                                          Resize(size=[image_height, image_width]),
                                          FlipVertical(p=0.5),
                                          FlipHorizontal(p=0.5),
                                          # Mean and std, obtained from the dataset
                                          Normalize(n=1, mean=[0.52096, 0.51698, 0.545980],
                                                          std=[0.10380, 0.11190, 0.118877])]
                                          )
            else:
                transformation = Compose([ToTensor(n=1),
                                          Affine(size=[0.5*image_height, 0.5*image_width],
                                                 scale=0.01*(m-1), p=0.5),
                                          Rotate(limit=[(m-1)*72,m*72], p=1.0),
                                          Resize(size=[image_height, image_width]),
                                          FlipVertical(p=0.5),
                                          FlipHorizontal(p=0.5),
                                          # Mean and std, obtained from the dataset
                                          Normalize(n=1, mean=[0.52096, 0.51698, 0.545980],
                                                          std=[0.10380, 0.11190, 0.118877])]
                                          )
            
            dataset_train_temp = RaabinDataset(image_dir=train_image_dir[n],
                                               transform=transformation)
            train_dataset = torch.utils.data.ConcatDataset([train_dataset, dataset_train_temp])
    
    # Splitting the dataset, if it is too large, to deminish if 'clip_valid'<1
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
    
    # Obtaining dataloader from dataset
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

def check_accuracy(loader, model, loss_fn, device='cuda' if torch.cuda.is_available() else 'cpu', **kwargs):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()
    # if title is passed, use it before 'Check acc' and 'Got an accuracy...'
    title = kwargs.get('title')
    if title==None: title = ''
    else: title = title+': '
    
    loop = tqdm(loader, desc=title+'Check acc')
    
    with torch.no_grad():
        for dictionary in loop:
            image, label = dictionary
            x, y = dictionary[image], dictionary[label]
            x, y = x.to(device=device), y.to(device=device)
            y = y.float()         
            pred = model(x)
            y = tf.center_crop(y, pred.shape[2:])
            pred = (pred > 0.5).float()
            loss = loss_fn(pred, y)
            num_correct += (pred == y).sum()
            num_pixels += torch.numel(pred)
            # next is to calculate dice-score
            pred = pred.to(device='cpu').to(torch.int32)
            y = y.to(device='cpu').to(torch.int32)
            dice = Dice(ignore_index=0)
            dice_score += dice(pred, y)
            loop.set_postfix(acc=str(round(100*num_correct.item()/int(num_pixels),4)))
            # deliting variables
            loss_item = loss.item()
            del loss, pred, x, y, image, label, dictionary
    # deliting variables
    num_correct_item = num_correct.item()
    num_pixels = int(num_pixels)
    dice_score_item = dice_score.item()
    len_loader = len(loader)
    del num_correct, dice_score, loader, loop
    
    print('\n'+title+f'Got an accuracy of {round(100*num_correct_item/int(num_pixels),4)}')
    
    print('\n'+title+f'Dice score: {round(dice_score_item/len_loader,4)}'+'\n')
    model.train()
    return 100*num_correct_item/num_pixels, loss_item, 100*dice_score_item/len_loader


def save_predictions_as_imgs(loader, model, folder='saved_images',
                             device='cuda' if torch.cuda.is_available() else 'cpu',
                             **kwargs):
    # If image is grayscale, if yes, we have to turn into rgb to save
    gray = kwargs.get('gray')
    # With model in evaluation
    model.eval()
    for idx, (dictionary) in enumerate(loader):
        image, label = dictionary
        x, y = dictionary[image], dictionary[label]
        x = x.to(device=device)
        y = y.to(dtype=torch.float32)
        y = y.to(device=device)
        with torch.no_grad():
            pred = model(x)
            y = tf.center_crop(y, pred.shape[2:])
            pred = (pred > 0.5).float()
        # If image is grayscale, transforming to 'rgb' (utils.save_image needs)
        if gray:
            pred = torch.cat([pred,pred,pred],1)
            y = y.unsqueeze(1)
            y = torch.cat([y,y,y],1)
            y = y.float()
            y = tf.center_crop(y, pred.shape[2:])
        save_image(pred, f'{folder}/pred_{idx}.png')
        save_image(y, f'{folder}/y_{idx}.png')
    
    model.train()
