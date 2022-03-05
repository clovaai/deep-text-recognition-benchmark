## This application initiates a TRBA model based on Deep-Text-Recognition-Framework for a single
## image inference. 

from trba.trba_detector import TRBADetector

import argparse
import numpy as np
from PIL import Image
import torch

    
#from dptr.trbaOcr import TrbaOCR 
saved_models = "/home/raki-dedigama/projects/rr/src/libraries/deep-text-recognition-benchmark/saved_models/TPS-ResNet-BiLSTM-Attn-Seed1111/MJ-ST/"
saved_model = saved_models + "best_accuracy.pth"


trba_detector = TRBADetector()
trba_detector.load_model_from_disk(saved_model)


image_path = 'demo_image/demo_1.png'
pil_image = Image.open(image_path)

#ean = trbaOCR.img_path_to_ean(image_path)
ean = trba_detector.predict(pil_image)
print("predicted : ", ean['pred'], ean['score'])

