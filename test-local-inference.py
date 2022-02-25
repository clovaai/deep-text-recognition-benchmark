## This application initiates a TRBA model based on Deep-Text-Recognition-Framework for a single
## image inference. 



import argparse
import numpy as np
from PIL import Image
import torch

    
#from dptr.trbaOcr import TrbaOCR 

saved_model = '/home/raki-dedigama/projects/rr/model-training/ean/models/trained-models/best_accuracy.pt'
#trbaOCR = TrbaOCR(device='cuda')    
#trbaOCR.load_model_from_disk(saved_model)

from dptr.trba_detector import TRBADetector
trba_detector = TRBADetector()
trba_detector.load_model_from_disk(saved_model)

# Run from image path
# pil_image = Image.open(image_path)#   
# predicted_text = trbaOCR.img_to_ean(pil_image)
# print("predicted_text : ", ean['pred'], ean['score'])

image_path = 'demo_image/ean.jpg'
pil_image = Image.open(image_path)

#ean = trbaOCR.img_path_to_ean(image_path)
ean = trba_detector.predict(pil_image)
print("predicted : ", ean['pred'], ean['score'])

# ## Trace model to torchscript        
# model = trbaOCR.model
# image = torch.rand(1,1,32,100).to('cuda')
# text_for_pred = torch.LongTensor(1, 26).fill_(0).to('cuda')

# traced_model = torch.jit.trace(model, (image, text_for_pred))
# print(traced_model)
# #scripted_model = torch.jit.script(model, (image, text_for_pred))
# #print(scripted_model)
# traced_model.save('traced_best_accuracy.pt')

# pred = traced_model(image,text_for_pred)
# print(pred.size())
# # Infer with traced model

# exit()


#trbaOCR.convert_model_to_onnx()



# ## Run from Pillow image
# pil_image = Image.open(image_path)#   
# predicted_text = trbaOCR.img_to_ean(pil_image)
# print("predicted_text : ", ean['pred'], ean['score'])

  


# ## Run from Pillow image
# pil_image = Image.open(image_path)#   
# predicted_text = trbaOCR.img_to_ean(pil_image)
# print("predicted_text : ", ean['pred'], ean['score'])