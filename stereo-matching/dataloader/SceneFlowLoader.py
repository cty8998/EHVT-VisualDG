import os
import torch
import torch.utils.data as data
import torch
import torchvision.transforms as transforms
import random
# from albumentations import Compose, OneOf
from PIL import Image, ImageOps
from . import preprocess 
from . import transforms
from .transforms import RandomColor
from . import readpfm as rp
import numpy as np
import cv2

import pdb

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def default_loader(path):
    return cv2.imread(path)
    # return Image.open(path).convert('RGB')


def disparity_loader(path):
    return rp.readPFM(path)


class ImageLoader_g(data.Dataset):
    def __init__(self, left, right,
                 left_disparity, left_occ, training, mode,
                 loader=default_loader, dploader=disparity_loader,
                 th=256, tw=512):

        self.left = left
        self.right = right
        self.disp_L = left_disparity
        self.occ_l = left_occ
        self.loader = loader
        self.dploader = dploader
        self.th = th
        self.tw = tw
        self.training = training
        self.mode = mode

    def __getitem__(self, index):
        batch = dict()

        left = self.left[index]
        right = self.right[index]
        disp_L = self.disp_L[index]
        disp_R = disp_L.replace('left', 'right')


        left_img = self.loader(left)
        right_img = self.loader(right)
        if self.mode != 'kitti':
            dataL, scaleL = self.dploader(disp_L)
            dataR, scaleR = self.dploader(disp_R)
            dataL = np.ascontiguousarray(dataL, dtype=np.float32)
            dataR = np.ascontiguousarray(dataR, dtype=np.float32)
        if self.mode == 'md':
            occ_L = Image.open(self.occ_l[index].replace('disp0GT.pfm', 'mask0nocc.png')).convert('L')
            occ_mask = np.ascontiguousarray(occ_L, dtype=np.float32)
        elif self.mode == 'eth':
            occ_mask = np.ascontiguousarray(Image.open(self.occ_l[index]))
        elif self.mode == 'kitti':
            dataL = Image.open(self.disp_L[index])
            dataL = np.ascontiguousarray(dataL, dtype=np.float32) / 256
            occ_mask = np.ascontiguousarray(Image.open(self.occ_l[index]))

        if disp_L.split('/')[-5] == 'flyingthings3d':
            dataL = -dataL
            dataR = -dataR

        # if self.training:
        #     left_img, right_img, dataL = horizontal_flip(left_img, right_img, dataL, dataR)

        #     h, w = left_img.shape[:2]

        #     x1 = random.randint(0, w - self.tw)
        #     y1 = random.randint(0, h - self.th)

        #     left_img = left_img[y1: y1 + self.th, x1: x1 + self.tw]
        #     right_img = right_img[y1: y1 + self.th, x1: x1 + self.tw]

        #     dataL = dataL[y1:y1 + self.th, x1:x1 + self.tw]

        #     img = {'left': left_img, 'right': right_img}
        #     # img = self.train_aug(img)

        #     left_img, right_img = img['left'], img['right']

        #     processed = preprocess.get_transform(augment=True)
        #     left_img = processed(left_img)
        #     right_img = processed(right_img)

        #     batch['imgL'], batch['imgR'], batch['disp_true'] = left_img, right_img, dataL

        #     return batch
        # else:
        processed = preprocess.get_transform(augment=False)
        left_img = processed(left_img)
        right_img = processed(right_img)

        batch['imgL'], batch['imgR'], batch['disp_true'], batch['occ_mask'] = left_img, right_img, dataL, occ_mask

        return batch

    def __len__(self):
        return len(self.left)
