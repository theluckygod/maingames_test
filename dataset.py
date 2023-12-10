import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset

import utils


class HeroDataset(Dataset):
    def __init__(self, data_path="hero_images", transform=None):
        self.data = os.listdir(data_path)
        self.data = list(filter(lambda x: x.endswith(".png"), self.data))
        self.data = sorted(self.data)
        self.data = list(map(utils.correct_label, self.data))
        
        hero_names = list(map(lambda x: x.split(".png")[0], self.data))
        self.label = dict(zip(hero_names, range(len(hero_names)))) 
        self.data_path = data_path

        self.transform = transform
    
    def __getitem__(self, index):
        filename = self.data[index]
        x = Image.open(os.path.join(self.data_path, filename))
        x = utils.process_train_image(x, wsize=128, hsize=128)
        label = self.label[filename.split(".png")[0]]

        if self.transform:
            x = self.transform(x)
        else:
            x = torch.from_numpy(np.array(x))
            x = x.permute(2, 0, 1)
        
        return x.to(torch.float32), torch.tensor(label)
    
    def __len__(self):
        return len(self.data)
    
    def get_class_names(self):
        return list(self.label.keys())

    def get_class_dict(self):
        return self.label
    

class HeroTestDataset(Dataset):
    def __init__(self, classes_dict, data_path="test_data/test_images", ground_truth_path="test_data/test.txt", transform=None):
        self.data_path = data_path
        with open(ground_truth_path, 'r') as f:
            lines = f.readlines()
            lines = [x.strip() for x in lines]
        self.ground_truth = list(map(lambda x: x.split('\t'), lines))
        self.classes_dict = classes_dict
        self.transform = transform

    def __getitem__(self, index):
        filepath = os.path.join(self.data_path, self.ground_truth[index][0])
        x = Image.open(filepath)
        x = utils.process_test_image(x, wsize=128, hsize=128)
        hero_name = self.ground_truth[index][1]
        assert hero_name in self.classes_dict, f"Hero name {hero_name} is not defined"
        label = self.classes_dict[hero_name]
        
        if self.transform:
            x = self.transform(x)
        else:
            x = torch.from_numpy(np.array(x))
            x = x.permute(2, 0, 1)

        return x.to(torch.float32), torch.tensor(label)

    def __len__(self):
        return len(self.ground_truth)
    

if __name__ == "__main__":
    train_set = HeroDataset()
    dataset = HeroTestDataset(classes_dict=train_set.get_class_dict())
    print(dataset[1])