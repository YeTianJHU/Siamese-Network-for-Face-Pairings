import torch
import os
from PIL import Image
from torch.utils.data.dataset import Dataset
import numpy as np
import cv2

class DatasetProcessing(Dataset):
    def __init__(self, data_folder, img_lst_path, transform=None):

        self.img_folder_path = data_folder
        self.transform = transform
        self.img_lst = self.read_split_file(img_lst_path)
 
        
    def __getitem__(self, index):

        img0_path = os.path.join(self.img_folder_path, self.img_lst[index][0])
        img1_path = os.path.join(self.img_folder_path, self.img_lst[index][1])

        label = map(float, self.img_lst[index][2])
        label = torch.from_numpy(np.array(label)).float()

        img0 = Image.open(img0_path)
        img1 = Image.open(img1_path)
        img0 = img0.convert('RGB')
        img1 = img1.convert('RGB')

        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)

        return img0, img1, label
        
    def __len__(self):
        return len(self.img_lst)

    def read_split_file(self, dir):
        res = []
        with open(dir) as f:
            for line in f:
                content = line.split()
                res.append(content)
        return res