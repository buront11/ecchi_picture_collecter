import csv
import ast

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms.transforms import RandomHorizontalFlip

# ecchi image dataset
class EIDataset(Dataset):
    def __init__(self, df, data_type):
        super(EIDataset, self).__init__()

        self.df = df
        self.image_paths = df['path'].values.tolist()

        # リストが文字列になるのでリストに再変換
        self.labels = torch.tensor([ast.literal_eval(d) for d in self.df['label']])
        if data_type == 'train':
            self.transform = transforms.Compose([
                # transforms.RandomHorizontalFlip(),
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])
        elif data_type == 'valid':
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        out_data = Image.open(self.image_paths[index]).convert('RGB')
        out_label = self.labels[index]

        if self.transform:
            out_data = self.transform(out_data)

        return out_data, out_label

if __name__=='__main__':
    a = EIDataset('./img/labels.csv')
    data, label = a[20]
    transform = transforms.ToPILImage()
    transform(data).show()
