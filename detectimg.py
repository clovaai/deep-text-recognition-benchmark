from detection import get_textbox
from imgproc import loadImage
from utils import group_text_box, get_image_list, eprint
import numpy as np
import cv2
import torch
import urllib.request
import os
from pathlib import Path

def detectimg(image, detector, text_threshold = 0.7, low_text = 0.4, link_threshold = 0.4, canvas_size = 2560, mag_ratio = 1., poly = False, gpu=True, imgH=64):
    '''
    Parameters:
    file: file path or numpy-array or a byte stream object
    '''

    if gpu is False:
        device = 'cpu'
        eprint('Using CPU. Note: This module is much faster with a GPU.')
    elif not torch.cuda.is_available():
        device = 'cpu'
        eprint('CUDA not available - defaulting to CPU. Note: This module is much faster with a GPU.')
    elif gpu is True:
        device = 'cuda'
    else:
        device = gpu

    if type(image) == str:
        img = loadImage(image)  # can accept URL
        if image.startswith('http://') or image.startswith('https://'):
            tmp, _ = urllib.request.urlretrieve(image)
            img_cv_grey = cv2.imread(tmp, cv2.IMREAD_GRAYSCALE)
            os.remove(tmp)
        else:
            img_cv_grey = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    
    elif type(image) == bytes:
        nparr = np.frombuffer(image, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_cv_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    elif type(image) == np.ndarray:
        img = image
        img_cv_grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    text_box = get_textbox(detector, img, canvas_size, mag_ratio, text_threshold,\
                            link_threshold, low_text, poly, device)
    horizontal_list, free_list = group_text_box(text_box, width_ths = 0.5, add_margin = 0.1)

    # should add filter to screen small box out

    image_list, max_width = get_image_list(horizontal_list, free_list, img_cv_grey, model_height = imgH)

    return image_list, max_width