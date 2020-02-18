import math
import random

from torch.utils.data import Dataset
import pandas as pd
import torch
import os
from skimage import io, transform
import skimage
from PIL import Image, ImageEnhance
import numpy as np
from torchvision.transforms import functional as F

import torch


class ChestXrayDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.labels_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.labels_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = os.path.join(self.root_dir, self.labels_frame.iloc[idx, 0])
        image = io.imread(img_name)
        if len(image.shape) < 3:  # RGB to gray
            image = skimage.color.gray2rgb(image)
        elif image.shape[-1] == 4:
            image = skimage.color.rgba2rgb(image)

        label = self.labels_frame.iloc[idx, 1]
        label = np.array(label)
        sample = (image, label)

        if self.transform:
            sample = self.transform(sample)
        return sample


class LeftToRightFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        image, label = sample
        if random.random() < self.p:
            image = Image.fromarray(image, mode='RGB')
            image = F.hflip(image)
            image = np.array(image)
        return image, label


class RandomRotation(object):
    def __init__(self, angle=10, p=0.5):
        self.angle = angle
        self.p = p

    def __call__(self, sample):
        image, label = sample
        if random.random() < self.p:
            image = Image.fromarray(image, mode='RGB')
            random_angle = random.random() * self.angle
            crop_width = int(image.size[0] / (1 + math.tan(math.radians(random_angle))) \
                             / math.cos(math.radians(random_angle)))
            image = image.rotate(random_angle, expand=1)
            image = F.center_crop(image, crop_width)
            image = np.array(image)
        return image, label


class ColorJitter(object):
    def __init__(self, p=0.5, color=1.0, contrast=1.0, brightness=1.0, sharpness=1.0):
        self.p = p
        self.color = color
        self.contrast = contrast
        self.brightness = brightness
        self.sharpness = sharpness

    def __call__(self, sample):
        image, label = sample
        if random.random() > self.p:
            return image, label
        image = Image.fromarray(image, mode='RGB')
        image = ImageEnhance.Color(image).enhance((random.random() + 0.5) * self.color)
        image = ImageEnhance.Contrast(image).enhance((random.random() + 0.5) * self.contrast)
        image = ImageEnhance.Brightness(image).enhance((random.random() + 0.5) * self.brightness)
        image = ImageEnhance.Sharpness(image).enhance((random.random() + 0.5) * self.sharpness)
        image = np.array(image)
        return image, label


class RandomCrop(object):
    def __init__(self, scale=200, p=0.5):
        self.scale = scale
        self.p = p

    def __call__(self, sample):
        image, label = sample

        if random.random() > self.p:
            return image, label

        origin_width = image.shape[0]
        if origin_width < self.scale:
            raise TypeError('scale should be smaller than origin width')

        random_scale = random.randint(self.scale, origin_width)
        random_start_left = random.randint(0, origin_width - random_scale)
        random_start_right = random.randint(0, origin_width - random_scale)
        new_image = image[random_start_left:random_start_left + random_scale,
                    random_start_right:random_start_right + random_scale,
                    :]

        return new_image, label


class Resize(object):
    def __init__(self, scale=224):
        self.scale = scale

    def __call__(self, sample):
        image, label = sample
        image = Image.fromarray(image, mode='RGB')
        image = image.resize((self.scale, self.scale), resample=Image.LANCZOS)
        image = np.array(image)
        return image, label


class ToTensor(object):
    def __call__(self, sample):
        image, label = sample
        image = image.transpose((2, 0, 1))
        return (torch.from_numpy(image).type(torch.cuda.FloatTensor),
                torch.from_numpy(label).type(torch.cuda.LongTensor))
