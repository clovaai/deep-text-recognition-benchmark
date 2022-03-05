## This script loads a pytorch model with file format .pth and 
## converts it to a torchscript model by tracing
## More about pytorch tracing : https://pytorch.org/tutorials/beginner/Intro_to_TorchScript_tutorial.html


import argparse
import numpy as np
from PIL import Image
import torch

from trba.trba_detector import TRBADetector


saved_models = "/home/raki-dedigama/projects/rr/src/libraries/deep-text-recognition-benchmark/saved_models/TPS-ResNet-BiLSTM-Attn-Seed1111/SMMs/"
saved_model_path = saved_models + "best_accuracy.pth"
traced_model_path = saved_models + "traced_best_accuracy.pt"
test_image_path = 'demo_image/demo_1.png'
test_image = Image.open(test_image_path)

# intialize client
trba_detector = TRBADetector()
trba_detector.load_model_from_disk(saved_model_path)
pred = trba_detector.predict(test_image)
print("predicted : ", pred['pred'], pred['score'])


## Trace model to torchscript with random inputs.    
print("Traced model with Torch Tracing")   
model = trba_detector.model
image = torch.rand(1,1,32,100).to('cuda')
text_for_pred = torch.LongTensor(1, 26).fill_(0).to('cuda')
is_train = False
traced_model = torch.jit.trace(model, (image, text_for_pred))
#print(traced_model)
pred = traced_model(image,text_for_pred)
print(pred.size())

traced_model.save(traced_model_path)




## Load traced model
print("Load traced model to TRBA Detector")
traced_model = torch.jit.load(traced_model_path)
#trba_detector_traced = TRBADetector()
#trba_detector_traced.load_model_from_disk(traced_model_path)
#pred = trba_detector_traced.predict(test_image)
#print("predicted : ", pred['pred'], pred['score'])

pred = traced_model(image,text_for_pred)
print(pred.size())



