## This script loads a pytorch model with file format .pth and 
## converts it to a torchscript model by tracing
## More about pytorch tracing : https://pytorch.org/tutorials/beginner/Intro_to_TorchScript_tutorial.html


import argparse
import numpy as np
from PIL import Image
import torch

from trba.core.trbaOcr import TrbaOCR 

saved_model = '/home/raki-dedigama/projects/rr/model-training/ean/models/trained-models/best_accuracy.pth'
trbaOCR = TrbaOCR()    
trbaOCR.load_model_from_disk(saved_model)

## Trace model to torchscript with random inputs.       
model = trbaOCR.model
image = torch.rand(1,1,32,100).to('cuda')
text_for_pred = torch.LongTensor(1, 26).fill_(0).to('cuda')

traced_model = torch.jit.trace(model, (image, text_for_pred))
print(traced_model)
traced_model.save('traced_best_accuracy.pt')

pred = traced_model(image,text_for_pred)
print(pred.size())

exit()


