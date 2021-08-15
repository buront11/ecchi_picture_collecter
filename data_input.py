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

# ecchi image dataset
class EIDataset(Dataset):
    def __init__(self, csv_path):
        super(EIDataset, self).__init__()

        df = pd.read_csv(csv_path)
        self.image_paths = df['path']
        # リストが文字列になるのでリストに再変換
        self.labels = [ast.literal_eval(d) for d in df['label']]
        self.transform = transforms.Compose(
            [transforms.Resize(256), transforms.Grayscale(), transforms.ToTensor()]
        )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        out_data = Image.open(self.image_paths[index])
        out_label = self.labels[index]

        if self.transform:
            out_data = self.transform(out_data)

        return out_data, out_label

if __name__=='__main__':
    a = EIDataset('./img/labels.csv')
    data, label = a[0]
    data.show()
