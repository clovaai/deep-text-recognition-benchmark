## This application initiates a TRBA model based on Deep-Text-Recognition-Framework for a single
## image inference. 



import argparse
import numpy as np
from PIL import Image


if __name__ == '__main__':
    
    from dptr.trbaOcr import TrbaOCR 
    
    saved_model = 'models/TPS-ResNet-BiLSTM-Attn.pth'
    trbaOCR = TrbaOCR(saved_model)    

    ## Run from image path
    image_path = 'demo_image/demo_1.png'
    ean = trbaOCR.img_path_to_ean(image_path)
    print("predicted : ", ean['pred'], ean['score'])
    ## Run from Pillow image
    pil_image = Image.open(image_path)#   
    predicted_text = trbaOCR.img_to_ean(pil_image)
    print("predicted_text : ", ean['pred'], ean['score'])