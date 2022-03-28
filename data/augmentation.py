import cv2
import random
from PIL import Image, ImageFilter

import numpy as np


def random_flat(img, min_ratio=0.7, max_ratio=1.33):
    '''
    输入是 PIL.Image
    '''
    ratio = random.random() * (max_ratio - min_ratio) + min_ratio
    w, h = img.size
    w = int(w * ratio)
    img = img.resize((w, h), Image.BICUBIC)
    return img

def random_flat_img_horizontal(img, min_ratio=0.7, max_ratio=1.33):
    '''
    输入是 nd.array
    '''
    ratio = random.random() * (max_ratio - min_ratio) + min_ratio
    img = cv2.resize(img, None, fx=ratio, fy=1., interpolation=cv2.INTER_CUBIC)
    return img

def random_blur(img, size=1):
    img = img.filter(ImageFilter.GaussianBlur(radius=size))
    return img