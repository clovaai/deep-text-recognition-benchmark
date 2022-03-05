
from trba.trba_triton_detector import TRBATritonDetector
from trba.trbaOcr import TrbaOCR

import argparse
import numpy as np
from PIL import Image


    
# read test image

image_path = 'demo_image/ean.jpg'
pil_image = Image.open(image_path)#   



import json
with open("triton.json") as f:
    triton_config = json.load(f)
ocr_component = "EANs"

triton_flags = triton_config[ocr_component]
trba_model_config = TrbaOCR()

# intialize client


#from trba.src.trba_triton_detector import TRBATritonDetector

trba_triton_detector = TRBATritonDetector(triton_flags = triton_flags)


trba_triton_detector.parse_trba_model()   

# preprocessed_batched_images = triton_client.
preds = trba_triton_detector.recognize_ocr(pil_image)
print("pred shape ", preds.shape)

ean = trba_triton_detector.post_process_preds(preds)
print("predicted : ", ean['pred'], ean['score'])