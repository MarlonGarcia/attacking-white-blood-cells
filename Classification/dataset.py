import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import torch
import pandas as pd


class RaabinDataset(Dataset):
    def __init__(self, image_dir, csv_file, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.names_labels = pd.read_csv(csv_file)
    
    def __len__(self):
        return len(self.names_labels)
    
    def classes(self):
        return torch.Tensor([0,1,2,3,4])
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        image_name = self.names_labels.iloc[idx, 0]
        image_path = os.path.join(self.image_dir, image_name)
        image = np.array(Image.open(image_path).convert('RGB'))
        label = torch.from_numpy(np.asarray(self.names_labels.iloc[idx, 1], int))
        dictionary = {'image0': image}
        # Since the functions work for dicionary, the image is a dictionary
        if self.transform is not None:
            dictionary = self.transform(dictionary)
        
        return dictionary, label