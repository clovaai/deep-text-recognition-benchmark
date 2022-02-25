import torch
import string
import argparse
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.nn.functional as F
import torch.onnx

from dptr.utils import CTCLabelConverter, AttnLabelConverter
from dptr.dataset import PillowImageDataset, RawDataset, AlignCollate, SingleImageDataset
from dptr.model import Model

from dptr.trbaOcr import TrbaOCR

class TRBADetector():

    def __init__(self):

        # initializes TRBA model with default options
        self.trba_ocr = TrbaOCR()
        self.opt  = self.trba_ocr.opt
        self.model = self.trba_ocr.model 
        self.converter = self.trba_ocr.converter          
        

    def load_model_from_disk(self, saved_model):
        print('loading pretrained model from %s' % saved_model)
        self.model = torch.load(saved_model)
        #self.model.load_state_dict(torch.load(opt.saved_model, map_location=device))
        print('Successfully loaded pretrained model')
       

    def img_to_image_loader(self, image):

        opt = self.opt

        AlignCollate_demo = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD)
                   
        if image is not None:
            
            image_data = PillowImageDataset(image, opt)
            image_loader = torch.utils.data.DataLoader(
                image_data, batch_size=opt.batch_size,
                shuffle=False,
                num_workers=int(opt.workers),
                collate_fn=AlignCollate_demo, pin_memory=True)

        else:
            print("Could not find image path for inference.")
        
        return image_loader


    # def img_path_to_ean(self, image_path):
       
       

    #     opt = self.opt

    #     """Single Image Dataset Preparation from Image Path """
    #     AlignCollate_demo = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD)
          
    #     if image_path is not None:
    #         image_data = SingleImageDataset(image_path, opt = opt)
    #         image_loader = torch.utils.data.DataLoader(
    #             image_data, batch_size=opt.batch_size,
    #             shuffle=False,
    #             num_workers=int(opt.workers),
    #             collate_fn=AlignCollate_demo, pin_memory=True)

    #     else:
    #         print("Could not find image path for inference.")

    #     return self.predict(image_loader)        


    # def img_to_ean(self, pillow_image):
       
    #     opt = self.opt

    #     AlignCollate_demo = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD)
                   
    #     if pillow_image is not None:
            
    #         image_data = PillowImageDataset(pillow_image, opt)
    #         image_loader = torch.utils.data.DataLoader(
    #             image_data, batch_size=opt.batch_size,
    #             shuffle=False,
    #             num_workers=int(opt.workers),
    #             collate_fn=AlignCollate_demo, pin_memory=True)

    #     else:
    #         print("Could not find image path for inference.")


        return self.predict(image_loader)

    def predict(self, image):
        
        image_loader = self.img_to_image_loader(image)
        opt = self.opt
             
        device = self.opt.device
    
        # predict
        self.model.eval()
        output = {}
        with torch.no_grad():
            for image_tensors, image_path_list in image_loader:
                batch_size = image_tensors.size(0)
               
                image = image_tensors.to(device)
                print("image shape " ,image.shape)
                # For max length prediction
                length_for_pred = torch.IntTensor([opt.batch_max_length] * batch_size).to(device)
                text_for_pred = torch.LongTensor(batch_size, opt.batch_max_length + 1).fill_(0).to(device)
                print("text for pred shape :", text_for_pred.size())
                preds = self.model(image, text_for_pred, is_train=False)
                print("pred shape ", preds.shape)
             

                # select max probabilty (greedy decoding) then decode index to character
                _, preds_index = preds.max(2)
                preds_str = self.converter.decode(preds_index, length_for_pred)

                                    
            
                preds_prob = F.softmax(preds, dim=2)
                preds_max_prob, _ = preds_prob.max(dim=2)
                for img_name, pred, pred_max_prob in zip(image_path_list, preds_str, preds_max_prob):
                    if 'Attn' in opt.Prediction:
                        pred_EOS = pred.find('[s]')
                        pred = pred[:pred_EOS]  # prune after "end of sentence" token ([s])
                        pred_max_prob = pred_max_prob[:pred_EOS]

                    # calculate confidence score (= multiply of pred_max_prob)
                    confidence_score = pred_max_prob.cumprod(dim=0)[-1]                  
                
                   
                    #print(f'\t{pred:25s}\t{confidence_score:0.4f}')
                    output['pred'] = pred
                    output['score'] = np.array(confidence_score.cpu())
                    
        
        return output