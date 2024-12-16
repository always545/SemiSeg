from .transform import *

from copy import deepcopy
import math
import numpy as np
import os
import random

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

def load_data(root,is_train):
    if is_train:
        filepath = os.path.join(root,'ImageSets','Segmentation','train.txt')
    else:
        filepath = os.path.join(root,'ImageSets','Segmentation','val.txt')
    with open(filepath,'r') as f:
        train_data = f.read().split()
    images = [os.path.join(root,'JPEGImages',i+'.jpg') for i in train_data]
    labels = [os.path.join(root,'SegmentationClass',i+'.png') for i in train_data]
    return images,labels

class SemiDataset(Dataset):
    def __init__(self, name, root, mode, size=None,lmg_path:list[str]=None,lbl_path=None,nsample=None):
        self.name = name
        self.root = root
        self.mode = mode
        self.size = size
        self.ids = lmg_path
        self.lbl = lbl_path


    def __getitem__(self, item):
        id = self.ids[item]
        img = Image.open(id).convert('RGB')
        mask = Image.fromarray(np.array(Image.open(self.lbl[item])))

        if self.mode == 'val':
            img, mask = normalize(img, mask)
            return img, mask, id

        img, mask = resize(img, mask, (0.5, 2.0))
        ignore_value = 254 if self.mode == 'train_u' else 255
        img, mask = crop(img, mask, self.size, ignore_value)
        img, mask = hflip(img, mask, p=0.5)

        if self.mode == 'train_l':
            return normalize(img, mask)

        img_w, img_s1, img_s2 = deepcopy(img), deepcopy(img), deepcopy(img)

        if random.random() < 0.8:
            img_s1 = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img_s1)
        img_s1 = transforms.RandomGrayscale(p=0.2)(img_s1)
        img_s1 = blur(img_s1, p=0.5)
        cutmix_box1 = obtain_cutmix_box(img_s1.size[0], p=0.5)

        if random.random() < 0.8:
            img_s2 = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img_s2)
        img_s2 = transforms.RandomGrayscale(p=0.2)(img_s2)
        img_s2 = blur(img_s2, p=0.5)
        cutmix_box2 = obtain_cutmix_box(img_s2.size[0], p=0.5)

        ignore_mask = Image.fromarray(np.zeros((mask.size[1], mask.size[0])))

        img_s1, ignore_mask = normalize(img_s1, ignore_mask)
        img_s2 = normalize(img_s2)

        mask = torch.from_numpy(np.array(mask)).long()
        ignore_mask[mask == 254] = 255

        return normalize(img_w), img_s1, img_s2, ignore_mask

    def __len__(self):
        return len(self.ids)


