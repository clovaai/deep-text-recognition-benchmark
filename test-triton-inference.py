

import argparse
from dptr.trbaOcr import TrbaOCR
import numpy as np
from PIL import Image
import torch

    
# read test image

image_path = 'demo_image/ean.jpg'
pil_image = Image.open(image_path)#   


## Read triton configs

import json
with open("triton.json") as f:
    triton_config = json.load(f)
ocr_component = "EANs"

triton_flags = triton_config[ocr_component]
trba_model_config = TrbaOCR()

# intialize client
from dptr.trba_triton_detector import TRBATritonDetector

trba_triton_detector = TRBATritonDetector(triton_flags = triton_flags, trba_model_config= trba_model_config)


trba_triton_detector.parse_trba_model()   

# preprocessed_batched_images = triton_client.
preds = trba_triton_detector.recognize_ocr(pil_image)
print("pred shape ", preds.shape)
