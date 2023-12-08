import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as F

import utils


class HeroDataset(Dataset):
    def __init__(self, data_path="hero_images", transform=True):
        self.data = os.listdir(data_path)
        self.data = list(filter(lambda x: x.endswith(".png"), self.data))
        hero_names = list(map(lambda x: x.split(".png")[0], self.data))
        self.label = dict(zip(hero_names, range(len(hero_names)))) 
        self.data_path = data_path

        self.transform = transform
    
    def __getitem__(self, index):
        filename = self.data[index]
        x = Image.open(os.path.join(self.data_path, filename))
        label = self.label[filename.split(".png")[0]]

        if self.transform:
            x = utils.process_test_image(x, wsize=32, hsize=32)
        
        x = torch.from_numpy(np.array(x))

        x = x.permute(2, 0, 1)
        
        return x.to(torch.float32), torch.tensor(label)
    
    def __len__(self):
        return len(self.data)
    
    def get_class_names(self):
        return list(self.label.keys())
    

if __name__ == "__main__":
    dataset = HeroDataset()
    print(dataset.get_class_names())
    print(dataset[1])