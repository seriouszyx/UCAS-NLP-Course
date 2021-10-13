import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, random_split
from torchvision import transforms


class MyData(Dataset):
    def __init__(self, filepath, transform=None):
        self.filepath = filepath
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize([224, 224]),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        else:
            self.transform = transform
        self.filepath_list = np.array([filepath + i for i in os.listdir(filepath)])

    def __getitem__(self, index):
        image = Image.open(self.filepath_list[index])
        image = self.transform(image)
        label = 0 if self.filepath_list[index].split('/')[-1].startswith('cat') else 1
        label = torch.as_tensor(label, dtype=torch.int64)
        return image, label

    def __len__(self):
        return len(self.filepath_list)


def dataset_split(full_ds, train_rate):
    train_size = int(train_rate * len(full_ds))
    test_size = len(full_ds) - train_size
    train_ds, test_ds = random_split(full_ds, [train_size, test_size])
    return train_ds, test_ds

