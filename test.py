import os
import time
import argparse

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import numpy as np
from torch_baidu_ctc import CTCLoss
from nltk.metrics.distance import edit_distance

from utils import CTCLabelConverter, AttnLabelConverter, Averager
from dataset import hierarchical_dataset, AlignCollate
from model import Model


def benchmark_all_eval(model, criterion, converter, opt, calculate_infer_time=False):
    """ evaluation with 10 benchmark evaluation datasets """
    list_accuracy = []
    Total_forward_time = 0
    Total_evaluation_data_number = 0
    # The evaluation datasets, dataset order is same with Table 1 in our paper.
    eval_data_list = ['IIIT5k_3000', 'SVT', 'IC03_860', 'IC03_867', 'IC13_857',
                      'IC13_1015', 'IC15_1811', 'IC15_2077', 'SVTP', 'CUTE80']

    if calculate_infer_time:
        evaluation_batch_size = 1  # batch_size should be 1 to calculate the GPU inference time per image.
    else:
        evaluation_batch_size = opt.batch_size

    print('-' * 80)
    for eval_data in eval_data_list:
        eval_data_path = os.path.join(opt.eval_data, eval_data)
        AlignCollate_evaluation = AlignCollate(imgH=opt.imgH, imgW=opt.imgW)
        eval_data = hierarchical_dataset(root=eval_data_path, opt=opt)
        print('-' * 80)
        Total_evaluation_data_number += len(eval_data)
        evaluation_loader = torch.utils.data.DataLoader(
            eval_data, batch_size=evaluation_batch_size,
            shuffle=False,
            num_workers=int(opt.workers),
            collate_fn=AlignCollate_evaluation, pin_memory=True)

        _, accuracy_by_best_model, _, _, _, infer_time = validation(
            model, criterion, evaluation_loader, converter, opt)
        Total_forward_time += infer_time
        list_accuracy.append(f'{accuracy_by_best_model:0.3f}')

    averaged_forward_time = Total_forward_time / Total_evaluation_data_number * 1000
    params_num = sum([np.prod(p.size()) for p in model.parameters()])

    evaluation_log = 'accuracy: '
    for name, accuracy in zip(eval_data_list, list_accuracy):
        evaluation_log += f'{name}: {accuracy}\t'
    evaluation_log += f'averaged_infer_time: {averaged_forward_time:0.3f}, # parameters: {params_num/1e6:0.3f}'
    print(evaluation_log)
    with open(f'./result/{opt.experiment_name}/log_all_evaluation.txt', 'a') as log:
        log.write(evaluation_log + '\n')

    return None


def validation(model, criterion, evaluation_loader, converter, opt):
    """ validation or evaluation """
    for p in model.parameters():
        p.requires_grad = False

    n_correct = 0
    norm_ED = 0
    max_length = 25
    length_of_data = 0
    infer_time = 0
    valid_loss_avg = Averager()

    for i, (cpu_images, cpu_texts) in enumerate(evaluation_loader):
        batch_size = cpu_images.size(0)
        length_of_data = length_of_data + batch_size
        with torch.no_grad():
            image = cpu_images.cuda()
            # For max length prediction
            length_for_pred = torch.cuda.IntTensor([max_length] * batch_size)
            text_for_pred = torch.cuda.LongTensor(batch_size, max_length + 1).fill_(0)

            text_for_loss, length_for_loss = converter.encode(cpu_texts)

        start_time = time.time()
        if 'CTC' in opt.Prediction:
            preds = model(image, text_for_pred)
            forward_time = time.time() - start_time

            # Calculate evaluation loss for CTC deocder.
            preds_size = torch.IntTensor([preds.size(1)] * batch_size)
            preds = preds.permute(1, 0, 2)  # to use CTCloss format
            cost = criterion(preds, text_for_loss, preds_size, length_for_loss) / batch_size

            # Select max probabilty (greedy decoding) then decode index to character
            _, preds = preds.max(2)
            preds = preds.transpose(1, 0).contiguous().view(-1)
            sim_preds = converter.decode(preds.data, preds_size.data)

        else:
            preds = model(image, text_for_pred, is_train=False)
            forward_time = time.time() - start_time

            preds = preds[:, :text_for_loss.shape[1] - 1, :]
            target = text_for_loss[:, 1:]  # without [GO] Symbol
            cost = criterion(preds.contiguous().view(-1, preds.shape[-1]), target.contiguous().view(-1))

            # select max probabilty (greedy decoding) then decode index to character
            _, preds_index = preds.max(2)
            sim_preds = converter.decode(preds_index, length_for_pred)
            cpu_texts = converter.decode(text_for_loss[:, 1:], length_for_loss)

        infer_time += forward_time
        valid_loss_avg.add(cost)

        # calculate accuracy.
        for pred, gt in zip(sim_preds, cpu_texts):
            if 'CTC' not in opt.Prediction:
                pred = pred[:pred.find('[s]')]  # prune after "end of sentence" token ([s])
                gt = gt[:gt.find('[s]')]

            if pred == gt:
                n_correct += 1
            norm_ED += edit_distance(pred, gt) / len(gt)

    accuracy = n_correct / float(length_of_data) * 100

    return valid_loss_avg.val(), accuracy, norm_ED, sim_preds, cpu_texts, infer_time


def test(opt):
    """ model configuration """
    if 'CTC' in opt.Prediction:
        converter = CTCLabelConverter(opt.character)
    else:
        converter = AttnLabelConverter(opt.character)
        
    if opt.rgb:
        opt.input_channel = 3
    
    opt.num_class = len(converter.character)
    model = Model(opt)
    print('model input parameters', opt.imgH, opt.imgW, opt.num_fiducial, opt.input_channel, opt.output_channel,
          opt.hidden_size, opt.num_class, opt.batch_max_length, opt.Transformation, opt.FeatureExtraction,
          opt.SequenceModeling, opt.Prediction)
    model = torch.nn.DataParallel(model).cuda()

    # load model
    if opt.saved_model != '':
        print('loading pretrained model from %s' % opt.saved_model)
        model.load_state_dict(torch.load(opt.saved_model))
        opt.experiment_name = '_'.join(opt.saved_model.split('/')[1:])
    # print(model)

    """ keep evaluation model and result logs """
    os.makedirs(f'./result/{opt.experiment_name}', exist_ok=True)
    os.system(f'cp {opt.saved_model} ./result/{opt.experiment_name}/')

    """ setup loss """
    if 'CTC' in opt.Prediction:
        criterion = CTCLoss(reduction='sum')
    else:
        criterion = torch.nn.CrossEntropyLoss(ignore_index=0).cuda()  # ignore [GO] token = ignore index 0

    """ evaluation """
    model.eval()
    if opt.benchmark_all_eval:  # evaluation with 10 benchmark evaluation datasets
        benchmark_all_eval(model, criterion, converter, opt)
    else:
        AlignCollate_evaluation = AlignCollate(imgH=opt.imgH, imgW=opt.imgW)
        eval_data = hierarchical_dataset(root=opt.eval_data, opt=opt)
        evaluation_loader = torch.utils.data.DataLoader(
            eval_data, batch_size=opt.batch_size,
            shuffle=False,
            num_workers=int(opt.workers),
            collate_fn=AlignCollate_evaluation, pin_memory=True)
        _, accuracy_by_best_model, _, _, _, _ = validation(
            model, criterion, evaluation_loader, converter, opt)

        print(accuracy_by_best_model)
        with open('./result/{0}/log_evaluation.txt'.format(opt.experiment_name), 'a') as log:
            log.write(str(accuracy_by_best_model) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_data', required=True, help='path to evaluation dataset')
    parser.add_argument('--benchmark_all_eval', action='store_true', help='evaluate 10 benchmark evaluation datasets')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
    parser.add_argument('--batch_size', type=int, default=192, help='input batch size')
    parser.add_argument('--saved_model', default='', help="path to saved_model to evaluation")
    """ Data processing """
    parser.add_argument('--batch_max_length', type=int, default=25, help='maximum-label-length')
    parser.add_argument('--imgH', type=int, default=32, help='the height of the input image')
    parser.add_argument('--imgW', type=int, default=100, help='the width of the input image')
    parser.add_argument('--rgb', action='store_true', help='use rgb input')
    parser.add_argument('--character', type=str, default='0123456789abcdefghijklmnopqrstuvwxyz', help='character label')
    parser.add_argument('--sensitive', action='store_true', help='for sensitive character mode')
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

    cudnn.benchmark = True
    cudnn.deterministic = True
    opt.num_gpu = torch.cuda.device_count()

    test(opt)
