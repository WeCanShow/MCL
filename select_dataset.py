import copy
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm
from PIL import Image

class SelectDataset(Dataset):

    def __init__(self, dataset, pre_idx=None, transform=None):

        self.root = dataset.root
        self.train = dataset.train
        self.target_transform = dataset.target_transform

        if pre_idx is None:
            self.data = copy.deepcopy(dataset.data)
            self.targets = np.array(copy.deepcopy(dataset.targets))
        else:
            self.data = copy.deepcopy(dataset.data[pre_idx])
            self.targets = np.array(copy.deepcopy(dataset.targets))[pre_idx]
        if transform is None:
            self.transform = dataset.transform
        else:
            self.transform = transform
        self.channels, self.width, self.height = self.__shape_info__()

    def __getitem__(self, item):
        img = self.data[item]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)

        label_idx = self.targets[item]
        label = np.zeros(10)
        label[label_idx] = 1 # 把num型的label变成10维列表。
        label = torch.Tensor(label)

        return img, label

    def __len__(self):
        return len(self.data)

    def __shape_info__(self):
        return self.data.shape[1:]


