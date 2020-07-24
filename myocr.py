import string
import argparse

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.nn.functional as F

from utils import CTCLabelConverter, AttnLabelConverter
from dataset import InferenceDataset, AlignCollate
from model import Model
from detectimg import detectimg
from detection import get_detector

from glob import glob
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def getDetector():
    DETECTOR_PATH = 'craft_mlt_25k.pth'
    detector = get_detector(DETECTOR_PATH, device)
    return detector

def myocr(opt):
    """ model configuration """
    if 'CTC' in opt.Prediction:
        converter = CTCLabelConverter(opt.character)
    else:
        converter = AttnLabelConverter(opt.character)
    opt.num_class = len(converter.character)

    if opt.rgb:
        opt.input_channel = 3

    model = Model(opt)
    print('model input parameters', opt.imgH, opt.imgW, opt.num_fiducial, opt.input_channel, opt.output_channel,
          opt.hidden_size, opt.num_class, opt.batch_max_length, opt.Transformation, opt.FeatureExtraction,
          opt.SequenceModeling, opt.Prediction)
    model = torch.nn.DataParallel(model).to(device)

    # load OCR model
    print('loading pretrained model from %s' % opt.saved_model)
    model.load_state_dict(torch.load(opt.saved_model, map_location=device))
    DETECTOR_PATH = 'craft_mlt_25k.pth'
    detector = get_detector(DETECTOR_PATH, device)

    # log
    log = open(f'./OCR_log.txt', 'a')
    dashed_line = '-' * 100
    head = f'{"box area":^35s}\t{"predicted_labels":25s}\tconfidence score'
    print(f'{dashed_line}\n{head}\n{dashed_line}')
    log.write(f'{dashed_line}\n{head}\n{dashed_line}\n')

    # OCR
    for image_dir in opt.image_folder:
        # detect image
        image_list, max_width = detectimg(image_dir,detector)
        coord = [item[0] for item in image_list]
        img_list = [item[1] for item in image_list]

        # prepare data. two demo images from https://github.com/bgshih/crnn#run-demo
        AlignCollate_demo = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD)
        demo_data = InferenceDataset(image_list=img_list, opt=opt)  # use InferenceDataset
        demo_loader = torch.utils.data.DataLoader(
            demo_data, batch_size=opt.batch_size,
            shuffle=False,
            num_workers=int(opt.workers),
            collate_fn=AlignCollate_demo, pin_memory=True)
        print(f'{image_dir}')
        log.write(f'{image_dir}\n')
        # predict
        model.eval()
        with torch.no_grad():
            for image_tensors, _ in demo_loader:
                batch_size = image_tensors.size(0)
                image = image_tensors.to(device)
                # For max length prediction
                length_for_pred = torch.IntTensor([opt.batch_max_length] * batch_size).to(device)
                text_for_pred = torch.LongTensor(batch_size, opt.batch_max_length + 1).fill_(0).to(device)

                if 'CTC' in opt.Prediction:
                    preds = model(image, text_for_pred)

                    # Select max probabilty (greedy decoding) then decode index to character
                    preds_size = torch.IntTensor([preds.size(1)] * batch_size)
                    _, preds_index = preds.max(2)
                    # preds_index = preds_index.view(-1)
                    preds_str = converter.decode(preds_index.data, preds_size.data)

                else:
                    preds = model(image, text_for_pred, is_train=False)

                    # select max probabilty (greedy decoding) then decode index to character
                    _, preds_index = preds.max(2)
                    preds_str = converter.decode(preds_index, length_for_pred)

                preds_prob = F.softmax(preds, dim=2)
                preds_max_prob, _ = preds_prob.max(dim=2)
                for i, zipped in enumerate(zip(preds_str, preds_max_prob)):
                    pred, pred_max_prob = zipped
                    if 'Attn' in opt.Prediction:
                        pred_EOS = pred.find('[s]')
                        pred = pred[:pred_EOS]  # prune after "end of sentence" token ([s])
                        pred_max_prob = pred_max_prob[:pred_EOS]

                    # calculate confidence score (= multiply of pred_max_prob)
                    confidence_score = pred_max_prob.cumprod(dim=0)[-1]
                    box = " ".join(map(str, coord[i]))
                    print(f'{box:35s}\t{pred:25s}\t{confidence_score:0.4f}')
                    log.write(f'{box:35s}\t{pred:25s}\t{confidence_score:0.4f}\n')
        print(dashed_line)
        log.write(f'{dashed_line}\n')
    log.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_folder', required=True, help='path to image_folder which contains text images')
    parser.add_argument('--infer_num',required=True,type=int,help='number of inference data')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
    parser.add_argument('--batch_size', type=int, default=192, help='input batch size')
    parser.add_argument('--saved_model', required=True, help="path to saved_model to evaluation")
    """ Data processing """
    parser.add_argument('--batch_max_length', type=int, default=25, help='maximum-label-length')
    parser.add_argument('--imgH', type=int, default=32, help='the height of the input image')
    parser.add_argument('--imgW', type=int, default=100, help='the width of the input image')
    parser.add_argument('--rgb', action='store_true', help='use rgb input')
    parser.add_argument('--character', type=str, default='0123456789abcdefghijklmnopqrstuvwxyz', help='character label')
    parser.add_argument('--sensitive', action='store_true', help='for sensitive character mode')
    parser.add_argument('--stat_dict', action='store_true', help='for sensitive character mode')
    parser.add_argument('--PAD', action='store_true', help='whether to keep ratio then pad for image resize')
    """ Model Architecture """
    parser.add_argument('--Transformation', type=str, required=True, help='Transformation stage. None|TPS')
    parser.add_argument('--FeatureExtraction', type=str, required=True, help='FeatureExtraction stage. VGG|RCNN|ResNet')
    parser.add_argument('--SequenceModeling', type=str, required=True, help='SequenceModeling stage. None|BiLSTM')
    parser.add_argument('--Prediction', type=str, required=True, help='Prediction stage. CTC|Attn')
    parser.add_argument('--num_fiducial', type=int, default=20, help='number of fiducial points of TPS-STN')
    parser.add_argument('--input_channel', type=int, default=1, help='the number of input channel of Feature extractor')
    parser.add_argument('--output_channel', type=int, default=512,
                        help='the number of output channel of Feature extractor')
    parser.add_argument('--hidden_size', type=int, default=256, help='the size of the LSTM hidden state')

    opt = parser.parse_args()

    """ vocab / character number configuration """
    if opt.sensitive:
        # opt.character = string.printable[:-6]  # same with ASTER setting (use 94 char).
        charlist = []
        with open('ko_char.txt', 'r', encoding='utf-8') as f:
            for c in f.readlines():
                charlist.append(c[:-1])
        opt.character = ''.join(charlist) + string.printable[:-38]

    if opt.stat_dict:
        opt.character = '0123456789'
        # charlist = []
        # with open('ko_char.txt', "r", encoding = "utf-8-sig") as f:
        #     for c in f.readlines():
        #         charlist.append(c[:-1])
        # number = '0123456789'
        # symbol  = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~ '
        # en_char = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
        # opt.character = number + symbol + en_char + ''.join(charlist)


    opt.image_folder = glob(opt.image_folder + '*')
    
    if len(opt.image_folder) > opt.infer_num:
        opt.image_folder = opt.image_folder[:opt.infer_num]

    cudnn.benchmark = True
    cudnn.deterministic = True
    opt.num_gpu = torch.cuda.device_count()

    myocr(opt)
