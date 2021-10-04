import string
import argparse

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.nn.functional as F

from deepTextRecognitionBenchmark.utils import CTCLabelConverter, AttnLabelConverter
from deepTextRecognitionBenchmark.dataset import RawDataset, AlignCollate, SingleImageDataset
from deepTextRecognitionBenchmark.model import Model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import argparse

class TrbaOCR:
    def __init__(self, saved_model):
       
        ## Argument parser carried forward as configuration data structure from deep-text-recongnition-benchmark. 
        parser = argparse.ArgumentParser()
        opt = parser.parse_args()
        
        opt.saved_model = saved_model


        opt.Transformation = 'TPS'
        opt.FeatureExtraction = 'ResNet'
        opt.SequenceModeling = 'BiLSTM'
        opt.Prediction = 'Attn'
        opt.workers = 4  
        opt.batch_size = 192

        opt.batch_max_length = 25
        opt.imgH = 32
        opt.imgW = 100
        opt.rgb = False
        opt.character = '0123456789abcdefghijklmnopqrstuvwxyz'
        opt.sensitive = False
        opt.PAD = False
        opt.num_fiducial = 20
        opt.input_channel = 1
        opt.output_channel = 512
        opt.hidden_size = 256


        if opt.sensitive:
            opt.character = string.printable[:-6]  # same with ASTER setting (use 94 char).
        cudnn.benchmark = True
        cudnn.deterministic = True
        opt.num_gpu = torch.cuda.device_count()
        self.opt = opt

    

    def img_to_ean(self, image_path):
  
        opt = self.opt
        """ model configuration """
        converter = AttnLabelConverter(opt.character)
        opt.num_class = len(converter.character)

        if opt.rgb:
            opt.input_channel = 3
        model = Model(opt)
        print('model input parameters', opt.imgH, opt.imgW, opt.num_fiducial, opt.input_channel, opt.output_channel,
            opt.hidden_size, opt.num_class, opt.batch_max_length, opt.Transformation, opt.FeatureExtraction,
            opt.SequenceModeling, opt.Prediction)
        model = torch.nn.DataParallel(model).to(device)

        # load model
        print('loading pretrained model from %s' % opt.saved_model)
        model.load_state_dict(torch.load(opt.saved_model, map_location=device))

        AlignCollate_demo = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD)
    
        #if opt.image_path is not None:
        if image_path is not None:
            image_data = SingleImageDataset(image_path, opt = opt)
            image_loader = torch.utils.data.DataLoader(
            image_data, batch_size=opt.batch_size,
            shuffle=False,
            num_workers=int(opt.workers),
            collate_fn=AlignCollate_demo, pin_memory=True)

        else:
            print("Could not find image path for inference.")
    


        # predict
        model.eval()
        with torch.no_grad():
            for image_tensors, image_path_list in image_loader:
                batch_size = image_tensors.size(0)
                print("batch size : ", batch_size)
                print("image tensor size :", image_tensors.shape)
                image = image_tensors.to(device)
                # For max length prediction
                length_for_pred = torch.IntTensor([opt.batch_max_length] * batch_size).to(device)
                text_for_pred = torch.LongTensor(batch_size, opt.batch_max_length + 1).fill_(0).to(device)

        
                preds = model(image, text_for_pred, is_train=False)

                # select max probabilty (greedy decoding) then decode index to character
                _, preds_index = preds.max(2)
                preds_str = converter.decode(preds_index, length_for_pred)

                print("preds ", preds.shape)

                log = open(f'./log_demo_result.txt', 'a')
                dashed_line = '-' * 80
                head = f'{"image_path":25s}\t{"predicted_labels":25s}\tconfidence score'
                
                print(f'{dashed_line}\n{head}\n{dashed_line}')
                log.write(f'{dashed_line}\n{head}\n{dashed_line}\n')

            

                preds_prob = F.softmax(preds, dim=2)
                preds_max_prob, _ = preds_prob.max(dim=2)
                for img_name, pred, pred_max_prob in zip(image_path_list, preds_str, preds_max_prob):
                    if 'Attn' in opt.Prediction:
                        pred_EOS = pred.find('[s]')
                        pred = pred[:pred_EOS]  # prune after "end of sentence" token ([s])
                        pred_max_prob = pred_max_prob[:pred_EOS]

                    # calculate confidence score (= multiply of pred_max_prob)
                    confidence_score = pred_max_prob.cumprod(dim=0)[-1]
                    
                    print(f'{img_name:25s}\t{pred:25s}\t{confidence_score:0.4f}')
                    log.write(f'{img_name:25s}\t{pred:25s}\t{confidence_score:0.4f}\n')

                log.close()
