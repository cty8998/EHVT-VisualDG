import os
import torch
import torch.utils.data as data
import torch
import torchvision.transforms as transforms
import random
from PIL import Image, ImageOps
from . import preprocess 
from . import listflowfile as lt
from . import readpfm as rp
import numpy as np

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def default_loader(path):
    return Image.open(path).convert('RGB')

def disparity_loader(path):
    return rp.readPFM(path)

class myImageFloder(data.Dataset):
    def __init__(self, left, right, left_disparity, training, envs_list=None, img_index_list=None, loader=default_loader, dploader= disparity_loader):
 
        self.left = left
        self.right = right
        self.disp_L = left_disparity
        self.loader = loader
        self.dploader = dploader
        self.training = training
        self.envs_list = envs_list
        self.img_index_list = img_index_list

    def __getitem__(self, index):
        left  = self.left[index]
        right = self.right[index]
        disp_L= self.disp_L[index]

        left_img = self.loader(left)
        try:
            right_img = self.loader(right)
        except:
            print(right)
        dataL, scaleL = self.dploader(disp_L)
        dataL = np.ascontiguousarray(dataL,dtype=np.float32)

        if self.training:  
            w, h = left_img.size
            th, tw = 256, 512

            x1 = random.randint(0, w - tw)
            y1 = random.randint(0, h - th)

            left_img = left_img.crop((x1, y1, x1 + tw, y1 + th))
            right_img = right_img.crop((x1, y1, x1 + tw, y1 + th))
            left_img_org = np.array(left_img)
            right_img_org = np.array(right_img)

            dataL = dataL[y1:y1 + th, x1:x1 + tw]

            processed = preprocess.get_transform(augment=False)  
            left_img   = processed(left_img)
            right_img  = processed(right_img)

            if self.envs_list != None:
                envs_id = self.envs_list[index]
                return left_img, right_img, dataL, left_img_org, right_img_org, envs_id
            elif self.img_index_list != None:
                img_index = self.img_index_list[index]
                return left_img, right_img, dataL, left_img_org, right_img_org, img_index
            else:
                return left_img, right_img, dataL, left_img_org, right_img_org
        else:
            processed = preprocess.get_transform(augment=False)  
            left_img       = processed(left_img)
            right_img      = processed(right_img) 
            return left_img, right_img, dataL

    def __len__(self):
        return len(self.left)
