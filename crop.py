import cv2
import numpy as np
import argparse
import string
import torch
import shutil
import time
import torch.backends.cudnn as cudnn
import os
import pytesseract
from tqdm import tqdm
import torch.utils.data
from torch.utils.data import DataLoader
from utils import CTCLabelConverter, AttnLabelConverter
from utils import extract_all_failed_imgaes
from dataset import AlignCollate, CropImageDataset
from model import Model
from my_utils import preprocess_image
import torch.nn.functional as F
# Disable GPU
os.environ["CUDA_VISIBLE_DEVICES"]=""
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')


def load_model(opt):
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

    # load model
    print('loading pretrained model from %s' % opt.saved_model)
    model.load_state_dict(torch.load(opt.saved_model, map_location=device))

    return model, converter


def predict(model, converter, image, opt):
    model.eval()
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # image = cv2.resize(image, (opt.imgW, opt.imgH))
    crop_data = CropImageDataset([image], opt)
    AlignCollate_crop = AlignCollate()
    crop_loader = DataLoader(crop_data,
                             batch_size=opt.batch_size,
                             shuffle=False,
                             collate_fn=AlignCollate_crop)
    for im_tensor, index in crop_loader:
        with torch.no_grad():
            batch_size = im_tensor.size(0)
            text_for_pred = torch.LongTensor(batch_size,
                                             opt.batch_max_length + 1).fill_(0).to(device)
            preds = model(im_tensor, text_for_pred)
            preds_size = torch.IntTensor([preds.size(1)] * batch_size)
            _, preds_index = preds.max(2)
            preds_str = converter.decode(preds_index, preds_size)
            preds_prob = F.softmax(preds, dim=2)
            preds_max_prob, _ = preds_prob.max(dim=2)
            return preds_str, preds_max_prob


def crop(opt, img_paths, ocr=False):
    model, converter = load_model(opt)
    failed = []
    for img_path in tqdm(img_paths):
        try:
            start_time = time.time()
            print('Input image path {}'.format(img_path))
            base_img_name = os.path.splitext(os.path.basename(img_path))[0]
            im = cv2.imread(img_path)
            ori_im = np.copy(im)
            ori_im_for_contours = np.copy(im)
            ori_im_copy = np.copy(ori_im)
            gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            # im = cv2.threshold(im, 127, 255, cv2.THRESH_BINARY_INV)[1]
            _, im = cv2.threshold(
            gray, 150, 255, cv2.THRESH_BINARY_INV)
            
            # Denoise
            im = cv2.medianBlur(im, 5)

            twelve_leads_names = ['i', 'avr', 'v1', 'v4',
                                'ii', 'avl', 'v2', 'v5',
                                'iii', 'avf', 'v3', 'v6']

            interested = {'avr': (0, 1), 'v1': (0, 2), 'v4': (0, 3),
                        'avl': (1, 1), 'v2': (1, 2), 'v5': (1, 3),
                        'avf': (2, 1), 'v3': (2, 2), 'v6': (2, 3),
                        'aur': (0, 1), 'u1': (0, 2), 'u4': (0, 3),
                        'aul': (1, 1), 'u2': (1, 2), 'u5': (1, 3),
                        'auf': (2, 1), 'u3': (2, 2), 'u6': (2, 3),
                        'avi': (1, 1),
                        'vi': (0, 2), 'iii': (2, 0)}

            twelve_leads = []
            for i in range(3):
                row = []
                for j in range(4):
                    row.append(None)
                twelve_leads.append(row)

            open_kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 1))
            dilate_kernel = cv2.getStructuringElement(cv2.MORPH_DILATE, (5, 3))
            opened = cv2.morphologyEx(
                im, cv2.MORPH_OPEN, open_kernel, iterations=0)
            dilated = cv2.morphologyEx(
                opened, cv2.MORPH_DILATE, dilate_kernel, iterations=5)
            _, contours, hierarchy = cv2.findContours(
                dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            time_after_morphology = time.time()

            print('Time taken for morphology : {}'.format(
                time_after_morphology - start_time))

            matched = set()
            for contour in contours:
                [x, y, w, h] = cv2.boundingRect(contour)
                cv2.rectangle(ori_im_for_contours, (x, y),
                            (x + w, y + h), (255, 0, 255), 2)

                if w / h > 5:
                    continue
                
                if w < 20 or h < 20:
                    continue

                if w > 500 or h > 500:
                    continue

                # draw rectangle around contour on original image
                cropped = im[y:y + h, x:x + w]
                # cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
                # _, thresh = cv2.threshold(cropped, 150, 255, cv2.THRESH_BINARY_INV)
                # cropped = thresh
                # cv2.imshow("Crop", cropped)
                # k = cv2.waitKey(0)
                # if k == ord('q'):
                #    exit() 
                # cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 255), 2)

                cropped_copy = np.copy(cropped)

                if not ocr:
                    preds_str, preds_max_prob = predict(model, converter, cropped, opt)
                    for pred, pred_max_prob in zip(preds_str, preds_max_prob):
                        confidence_score = pred_max_prob.cumprod(dim=0)[-1]
                        cv2.putText(ori_im_for_contours, str(pred) + str(confidence_score.item()), (x, y),
                                    cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 255))
                        # print('Prediction : {} Confidence : {}'.format(pred, confidence_score))
                        # cv2.imshow("Crop", cropped)
                        # k = cv2.waitKey(0)
                        # if k == ord('q'):
                        #     exit()
                        if pred in interested:
                            matched.add(pred)
                            row, col = interested[pred]
                            twelve_leads[row][col] = (int(x), int(y + h / 2))
                            # cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 255), 2)
                            # cv2.rectangle(ori_im, (x, y), (x + w, y + h), (255, 0, 255), 2)
                else:
                    config = '-psm 7'
                    pred = pytesseract.image_to_string(cropped, config=config)
                    cv2.putText(ori_im_for_contours, str(pred), (x, y),
                                cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 255))
                    if pred in interested:
                        matched.add(pred)
                        row, col = interested[pred]
                        twelve_leads[row][col] = (int(x), int(y + h / 2))

            time_after_detection = time.time()

            print('Time taken for detection : {}'.format(
                time_after_detection - time_after_morphology))

            print('Matched : {}'.format(matched))
            # Get horizontal distance:
            # Find any two leads that are not in the same column
            hor_interval = None
            word_1 = None
            word_2 = None
            for col in range(4):
                for row in range(3):
                    if twelve_leads[row][col] is not None:
                        if word_1 is not None:
                            word_2 = (row, col)
                            break
                        else:
                            word_1 = (row, col)
                            break
                if word_1 is not None and word_2 is not None:
                    assert word_1[1] != word_2[1]  # Not in the same column!
                    word_1_x = twelve_leads[word_1[0]][word_1[1]][0]
                    word_2_x = twelve_leads[word_2[0]][word_2[1]][0]
                    hor_interval = (word_2_x - word_1_x) / (word_2[1] - word_1[1])
                    break
            print('Horizontal interval is {}'.format(hor_interval))
            
            if hor_interval is None:
                print('Calculate hor interval by brute force now')
                

            vert_interval = None
            word_1 = None
            word_2 = None
            for row in range(3):
                for col in range(4):
                    if twelve_leads[row][col] is not None:
                        if word_1 is not None:
                            word_2 = (row, col)
                            break
                        else:
                            word_1 = (row, col)
                            break
                if word_1 is not None and word_2 is not None:
                    assert word_1[0] != word_2[0]  # Not in the same row!
                    word_1_y = twelve_leads[word_1[0]][word_1[1]][1]
                    word_2_y = twelve_leads[word_2[0]][word_2[1]][1]
                    vert_interval = (word_2_y - word_1_y) / (word_2[0] - word_1[0])
                    break
            print('Vertical interval is {}'.format(vert_interval))

            if hor_interval is None or vert_interval is None:
                print('Not able to calculate intervals between leads!')
                # Save error log(?)
                error_output_folder = os.path.join('./failed', base_img_name)
                if os.path.exists(error_output_folder):
                    shutil.rmtree(error_output_folder)
                os.makedirs(error_output_folder)
                cv2.imwrite(
                    '{}/{}.png'.format(error_output_folder, base_img_name), ori_im)
                cv2.imwrite('{}/binarised.png'.format(error_output_folder), im)
                cv2.imwrite('{}/contours.png'.format(error_output_folder),
                            ori_im_for_contours)
                cv2.imwrite('{}/opened.png'.format(error_output_folder), opened)
                cv2.imwrite('{}/dilated.png'.format(error_output_folder), dilated)
                failed.append(base_img_name)
            else:
                output_folder = os.path.join('./output', base_img_name)
                if os.path.exists(output_folder):
                    shutil.rmtree(output_folder)
                os.makedirs(output_folder)
                print('Horizontal interval {} Vertical interval {}'.format(
                    hor_interval, vert_interval))
                # Use the first non None element as anchor
                anchor = None
                for i in range(3):
                    if anchor is not None:
                        break
                    for j in range(4):
                        if twelve_leads[i][j] is not None:
                            anchor = (i, j)
                            break

                if anchor is None or hor_interval is None or vert_interval is None:
                    print('No anchor!')
                else:
                    anchor_row = anchor[0]
                    anchor_col = anchor[1]
                    print('Anchor is {}'.format(
                        twelve_leads_names[anchor_row * 4 + anchor_col]))
                    anchor_x, anchor_y = twelve_leads[anchor_row][anchor_col]
                    for i in range(3):
                        for j in range(4):
                            if twelve_leads[i][j] is None:
                                row_diff = i - anchor_row
                                col_diff = j - anchor_col
                                twelve_leads[i][j] = (int(anchor_x + col_diff * hor_interval),
                                                    int(anchor_y + row_diff * vert_interval))
                            lead_name = twelve_leads_names[i * 4 + j]
                            name_x, name_y = twelve_leads[i][j]
                            crop_x = name_x
                            crop_y = int(name_y - vert_interval / 2)
                            cropped = ori_im[crop_y:crop_y + int(vert_interval),
                                            crop_x:crop_x + int(hor_interval)]
                            cv2.circle(ori_im_copy, (name_x, name_y),
                                    5, (0, 0, 255), thickness=-1)
                            cv2.rectangle(ori_im_copy, (crop_x, crop_y),
                                        (crop_x + int(hor_interval),
                                        crop_y + int(vert_interval)),
                                        (0, 255, 0))
                            cv2.imwrite(
                                '{}/{}.png'.format(output_folder, lead_name), cropped)
                            # cv2.circle(ori_im, twelve_leads[i][j], 100, (255, 0, 0))

                # cv2.imwrite('./opened.png', opened)
                # cv2.imwrite('./dilated.png', dilated)
                # Save images
                cv2.imwrite('{}/binarised.png'.format(output_folder), im)
                cv2.imwrite('{}/output.png'.format(output_folder), ori_im)
                cv2.imwrite('{}/with_boxes.png'.format(output_folder), ori_im_copy)
                cv2.imwrite('{}/contours.png'.format(output_folder),
                            ori_im_for_contours)
                cv2.imwrite('{}/opened.png'.format(output_folder), opened)
                cv2.imwrite('{}/dilated.png'.format(output_folder), dilated)
        except Exception as e:
            print("Exception {}".format(e))
            failed.append(base_img_name)

    print('{} images failed!'.format(len(failed)))
    print('The failed ones are {}'.format(failed))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int,
                        help='number of data loading workers', default=4)
    parser.add_argument('--batch_size', type=int,
                        default=192, help='input batch size')
    parser.add_argument('--saved_model', default='./TPS-ResNet-BiLSTM-CTC.pth',
                        help="path to saved_model to evaluation")
    """ Data processing """
    parser.add_argument('--batch_max_length', type=int,
                        default=25, help='maximum-label-length')
    parser.add_argument('--imgH', type=int, default=32,
                        help='the height of the input image')
    parser.add_argument('--imgW', type=int, default=100,
                        help='the width of the input image')
    parser.add_argument('--rgb', action='store_true', help='use rgb input')
    parser.add_argument('--character', type=str,
                        default='0123456789abcdefghijklmnopqrstuvwxyz', help='character label')
    parser.add_argument('--sensitive', action='store_true',
                        help='for sensitive character mode')
    parser.add_argument('--PAD', action='store_true',
                        help='whether to keep ratio then pad for image resize')
    """ Model Architecture """
    parser.add_argument('--Transformation', type=str,
                        default='TPS', help='Transformation stage. None|TPS')
    parser.add_argument('--FeatureExtraction', type=str, default='ResNet',
                        help='FeatureExtraction stage. VGG|RCNN|ResNet')
    parser.add_argument('--SequenceModeling', type=str,
                        default='BiLSTM', help='SequenceModeling stage. None|BiLSTM')
    parser.add_argument('--Prediction', type=str,
                        default='CTC', help='Prediction stage. CTC|Attn')
    parser.add_argument('--num_fiducial', type=int, default=20,
                        help='number of fiducial points of TPS-STN')
    parser.add_argument('--input_channel', type=int, default=1,
                        help='the number of input channel of Feature extractor')
    parser.add_argument('--output_channel', type=int, default=512,
                        help='the number of output channel of Feature extractor')
    parser.add_argument('--hidden_size', type=int, default=256,
                        help='the size of the LSTM hidden state')
    parser.add_argument('--input_folder', type=str, default='/home/bowen/datasets/ecg/png_final_August')

    opt = parser.parse_args()

    """ vocab / character number configuration """
    if opt.sensitive:
        # same with ASTER setting (use 94 char).
        opt.character = string.printable[:-6]

    cudnn.benchmark = True
    cudnn.deterministic = True
    opt.num_gpu = torch.cuda.device_count()
    # opt.num_gpu = 0

    start_time = time.time()

    input_folder = opt.input_folder

    input_images = sorted([os.path.join(input_folder, image_name)
       for image_name in os.listdir(input_folder)])
    
    # input_images = [os.path.join(input_folder, image_name)
                    # for image_name in extract_all_failed_imgaes()]
    # input_images = ['./input/CHS004_Redacted.png']
    # input_images = ['./input/CHS004_Redacted.png']
    # input_images = input_images[:1]
    print(input_images)
    crop(opt, input_images)
    # model, converter = load_model(opt)
    # v5 = preprocess_image('/home/bowen/Pictures/v5.png')
    # cv2.imshow('v5', v5)
    # cv2.waitKey(0)
    # print(predict(model, converter, v5, opt))

    # interested_imgs = ['CHS004_Redacted', 'CHS005_Redacted', 'CHS053_Redacted', 'CHS085_Redacted', 'CHS085_Redacted1', 'CHS090_Redacted', 'CHS099_Redacted', 'CHS099_Redacted1', 'CHS103_Redacted', 'CHS119_Redacted', 'CHS136_Redacted', 'CHS137_Redacted', 'CHS162_Redacted', 'CHS179_Redacted', 'CHS192_Redacted', 'CHS197_Redacted', 'CHS206_Redacted', 'CHS207_Redacted', 'CHS227_Redacted', 'CHS231_Redacted', 'CHS247_Redacted', 'CHS253_Redacted', 'CHS262_Redacted', 'CHS289_Redacted', 'CHS306_Redacted', 'CHS311_Redacted', 'CHS323_Redacted', 'CHS329_Redacted', 'CHS330_Redacted', 'CHS332_Redacted', 'CHS336_Redacted', 'CHS338_Redacted', 'CHS341_Redacted', 'CHS350_Redacted', 'CHS352_Redacted', 'CHS359_Redacted', 'CHS361_Redacted', 'CHS362_Redacted', 'CHS419_Redacted', 'CHS420_Redacted', 'CHS449_Redacted', 'CHS506_Redacted', 'CHS508_Redacted', 'CHS537_Redacted', 'CHS554_Redacted', 'CHS555_Redacted', 'CHS557_Redacted', 'CHS560_Redacted', 'CHS571_Redacted', 'CHS601_Redacted', 'CHS602_Redacted', 'CHS613_Redacted', 'CHS620_Redacted']
    # # for img_path in tqdm(sorted(os.listdir(input_folder))):
    # input_images = sorted([os.path.join(input_folder, image_name + '.png') for image_name
    #                        in interested_imgs])
    # print(input_images)
    # crop(opt, input_images)
    #     full_img_path = os.path.join(input_folder, img_path)
    #     base_img_name = os.path.splitext(os.path.basename(full_img_path))[0]
    # crop(opt, full_img_path)

    # crop(opt)
    # end_time = time.time()
    # print('Time taken is {}'.format(end_time - start_time))
    # img = cv2.imread('./demo_image/avr.png')
    # model, converter = load_model(opt)
    # predict(model, converter, img, opt)
