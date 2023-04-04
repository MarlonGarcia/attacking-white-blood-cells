import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import torch


class RaabinDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_names = os.listdir(image_dir)
        basename = os.path.basename(image_dir)
        self.label_dir = os.path.join(os.path.dirname(os.path.dirname(image_dir)),
                                      'Ground Truth', basename)
    
    def __len__(self):
        return len(self.image_names)
        
    def classes(self):
        return torch.Tensor([0,1,2,3,4])
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        image = np.array(Image.open(os.path.join(self.image_dir, self.image_names[idx])).convert('RGB'))
        label1 = np.array(Image.open(os.path.join(self.label_dir, self.image_names[idx])).convert('RGB'))
        # to use just three conditions, we create another label with np.zeros
        label = np.zeros(np.shape(label1), np.uint8)
        label[:,:,0][label1[:,:,0]<50] = 1
        label[:,:,1][(label1[:,:,0]>50) & (label1[:,:,0]<200)] = 1
        label[:,:,2][label1[:,:,0]>200] = 1
        
        dictionary = {'image0': image, 'image1': label}
        
        if self.transform is not None:
            dictionary = self.transform(dictionary)
    
        return dictionary
