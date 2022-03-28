import argparse
import random
import time

import torch
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.optim as optim
from torch.utils.data import Dataloader, ConcatDataSet, Subset
import numpy as np
import torch.nn.functional as F
from nltk.metrics.distance import edit_distance

from model import MyModel
from collections import namedtuple
from utils import CTCLabelConverter, AttnLabelConverter, Averager
# from dataset import hierarchical_dataset, AlignCollate, Batch_Balanced_Dataset

from data.my_dataset import MyLmdbDataset


def get_all_characters():
    provinces = ["京", "津", "冀", "晋", "蒙", "辽", "吉", "黑", "沪",
                 "苏", "浙", "皖", "闽", "赣", "鲁", "豫", "鄂", "湘",
                 "粤", "桂", "琼", "渝", "川", "贵", "云", "藏", "陕",
                 "甘", "青", "宁", "新"]
    all_characters = "".join(provinces) + "#abcdefghjklmnpqrstuvwxyz0123456789警港澳学领使挂"
    return all_characters


def get_model():
    arch_dict = {
        "trans": None,
        "feat": "CRNN",
        "seq": "BILSTM",
        "head": "CTC"
    }
    feat_dict = {
        "input_c": 3,
        "output_c": 512
    }
    bilstm_dict = {
        "hidden_size": 512,
    }
    head_dict = {
        "num_classes": 74,  # CTC+1, Attn+2?
    }
    model = MyModel(arch_dict, feat_dict=feat_dict, bilstm_dict=bilstm_dict, head_dict=head_dict)

    # weight initialization
    for name, param in model.named_parameters():
        if 'localization_fc2' in name:
            print(f'Skip {name} as it is already initialized')
            continue
        try:
            if 'bias' in name:
                init.constant_(param, 0.0)
            elif 'weight' in name:
                init.kaiming_normal_(param)
        except Exception as e:  # for batchnorm.
            if 'weight' in name:
                param.data.fill_(1)
            continue

    filtered_parameters = []
    params_num = []
    for p in filter(lambda p: p.requires_grad, model.parameters()):
        filtered_parameters.append(p)
        params_num.append(np.prod(p.size()))
    print('Trainable params num : ', sum(params_num))
    [print(name, p.numel()) for name, p in
     filter(lambda p: p[1].requires_grad, model.named_parameters())]

    return model, filtered_parameters


def get_dataloader(dataset, is_train, batch_size=128, num_workers=16):
    dataloader = Dataloader(
        dataset, batch_size=batch_size, shuffle=is_train, num_workers=num_workers,
        pin_memory=is_train, drop_last=is_train
    )
    return dataloader


def main():
    base_lr = 1e-1
    experiment_name = ""
    save_root = ""

    model, train_parameters = get_model()
    optimizer = torch.optim.Adam(train_parameters, lr=base_lr, betas=(0., 0.999))
    scheduler = None
    criterion = torch.nn.CTCLoss(zero_infinity=True).cuda()
    all_characters = get_all_characters()
    converter = CTCLabelConverter(all_characters)

    val_dataset = MyLmdbDataset(
        root="/home/dl/liyunfei/project/rec_lmdb_dataset/test_db",
        db_name="cennavi_v1",
        max_length=100,
        all_characters=all_characters,
        input_c=3, input_h=32, input_w=160,
        mean=.5, std=.5,
        do_trans=False
    )
    val_dataloader = get_dataloader(val_dataset, False, batch_size=32, num_workers=8)

    loss_avg = Averager()
    start_time = time.time()
    best_accuracy = -1
    best_ned = 1e+6

    epochs = 200
    for epoch in range(epochs):
        train_loss = train(model, optimizer, scheduler, criterion, converter)
        valid_loss, current_accuracy, current_ned, preds, labels, infer_time, length_of_data = \
            val(model, criterion, val_dataloader, converter)
        elapsed_time = time.time() - start_time
        print(f'end one epoch --> [{epoch}/{epochs}] train Loss: {train_loss:0.5f} elapsed_time: {elapsed_time:0.5f}')

        for pred, gt in zip(preds[:5], labels[:5]):
            print(f'{pred:20s}, gt: {gt:20s},   {str(pred == gt)}')
        valid_log = f'[{epoch}/{epochs}] valid loss: {valid_loss:0.5f}'
        valid_log += f' accuracy: {current_accuracy:0.3f}, norm_ED: {current_ned:0.2f}'
        print(valid_log)

        if current_accuracy > best_accuracy:
            best_accuracy = current_accuracy
            torch.save(model.state_dict(), f'./saved_models/{experiment_name}/best_accuracy.pth')
        if current_ned < best_ned:
            best_ned = current_ned
            torch.save(model.state_dict(), f'./saved_models/{experiment_name}/best_norm_ED.pth')
        best_model_log = f'best_accuracy: {best_accuracy:0.3f}, best_norm_ED: {best_ned:0.2f}'
        print(best_model_log)

        net_save_path = f"{save_root}/experiment_name-{epoch}-{valid_loss:.4f}.p"
        state = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            # 'config': CONFIG,
            'loss': loss_avg.val(),
        }
        torch.save(state, net_save_path)


def train(model, optimizer, scheduler, criterion, converter):
    all_characters = get_all_characters()
    batch_size = 128
    num_workers = 16
    grad_clip = 5.
    input_w, input_h, input_c = 160, 32, 3
    mean, std = 0.5, 0.5

    model.train()

    dataset_synth = MyLmdbDataset(
        "/home/dl/liyunfei/project/rec_lmdb_dataset/train_val_db", "synth_v1",
        10, all_characters, input_w, input_h, input_c, mean, std, do_trans=True
    )  # 20w

    dataset_ccpd = MyLmdbDataset(
        "/home/dl/liyunfei/project/rec_lmdb_dataset/train_val_db", "ccpd",
        10, all_characters, input_w, input_h, input_c, mean, std, do_trans=True
    )  # 35w
    dataset_synth = Subset(dataset_synth, random.choices(range(0, dataset_synth.num_samples), k=100000))
    dataset_ccpd = Subset(dataset_ccpd, random.choices(range(0, dataset_ccpd.num_samples), k=200000))
    dataset = ConcatDataSet([dataset_ccpd, dataset_synth])
    dataloader = get_dataloader(dataset, True, batch_size=batch_size, num_workers=num_workers)

    loss_avg = Averager()
    for i, (input_batch, label_batch) in enumerate(dataloader):
        input_batch = input_batch.cuda()
        label_index, label_length = converter.encode(label_batch)

        pred_score_map = model(input_batch)
        pred_score_map = F.log_softmax(pred_score_map, dim=2)
        _, pred_index = pred_score_map.max(dim=2)

        device = pred_score_map.device
        pred_length = torch.LongTensor([pred_length.size(1)] * pred_score_map.size(0)).to(device)
        label_index = label_index.long().to(device)
        label_length = label_length.long().to(device)
        loss = criterion(pred_score_map, label_index, pred_length, label_length)

        model.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        loss_avg.add(loss.data)
        if i % 20 == 0:
            print(f"[{i}/{len(dataloader)}], loss: {loss_avg.val()}")
            loss_avg.reset()
    return loss_avg.val()


def val(model, criterion, dataloader, converter):
    batch_max_length = 10

    model.eval()

    n_correct = 0
    ned = 0
    length_of_data = 0
    infer_time = 0
    valid_loss_avg = Averager()
    with torch.no_grad():
        for i, (input_batch, label_batch) in enumerate(dataloader):
            batch_size = input_batch.size(0)
            length_of_data = length_of_data + batch_size
            input_batch = input_batch.cuda()

            # length_for_pred = torch.IntTensor([batch_max_length] * batch_size).cuda()
            # text_for_pred = torch.LongTensor(batch_size, batch_max_length + 1).fill_(0).cuda()

            text_for_loss, length_for_loss = converter.encode(label_batch, batch_max_length=batch_max_length)
            start_time = time.time()
            preds = model(input_batch, None).log_softmax(2)
            forward_time = time.time() - start_time

            # Calculate evaluation loss for CTC deocder.
            preds_size = torch.IntTensor([preds.size(1)] * batch_size)
            preds = preds.permute(1, 0, 2)  # to use CTCloss format
            cost = criterion(preds, text_for_loss, preds_size, length_for_loss)

            # Select max probabilty (greedy decoding) then decode index to character
            _, preds_index = preds.max(2)
            preds_index = preds_index.transpose(1, 0).contiguous().view(-1)
            preds_str = converter.decode(preds_index.data, preds_size.data)

            infer_time += forward_time
            valid_loss_avg.add(cost)

            # calculate accuracy.
            for pred, gt in zip(preds_str, label_batch):
                if pred == gt:
                    n_correct += 1
                if len(gt) == 0:
                    ned += 1
                else:
                    ned += edit_distance(pred, gt) / len(gt)
    accuracy = n_correct / float(length_of_data) * 100
    return valid_loss_avg.val(), accuracy, ned, preds_str, label_batch, infer_time, length_of_data

if __name__ == "__main__":
    main()