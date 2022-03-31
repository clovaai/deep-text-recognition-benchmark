import argparse

import torch
from torch.utils.data import  DataLoader

from model import MyModel
from data.my_dataset import TestLmdbDataset, TestAlignCollate
from utils import CTCLabelConverter, get_char_acc, get_line_acc


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pth_path", default="", help="path to pytorch checkpoint.")
    parser.add_argument("--min_w", type=float, default=60, help="min width of test image.")
    parser.add_argument("--score_t", type=float, default=0.9, help="filter threshold of recognize result.")
    parser.add_argument("--input_c", type=int, default=3)
    parser.add_argument("--input_h", type=int, default=32)
    parser.add_argument("--input_w", type=int, default=160)
    return parser.parse_args()

def main():
    args = get_args()
    args.pth_path = "/home/dl/liyunfei/project/lp_dev_v1/licenseplate_recognition/recog_dev/output/baseline/pth/best_char_acc_84.7117-epoch:178.pth"
    args.pth_path = "/home/dl/liyunfei/project/lp_dev_v1/licenseplate_recognition/recog_dev/output/baseline/pth/best_line_acc_52.9973-epoch:103.pth"
    args.input_c = 3
    args.input_h = 32
    args.input_w = 160

    arch_dict = {
        "trans": "None",
        "feat": "CRNN",
        "seq": "BILSTM",
        "head": "CTC"
    }
    feat_dict = {
        "input_c": 3,
        "output_c": 512
    }
    bilstm_dict = {
        "hidden_size": 256,
    }
    head_dict = {
        "num_classes": 74,  # CTC+1, Attn+2?
    }
    model = MyModel(arch_dict, feat_dict=feat_dict, bilstm_dict=bilstm_dict, head_dict=head_dict)
    model.load_state_dict(torch.load(args.pth_path, map_location="cpu"), strict=True)
    model.eval()
    model.cuda()

    all_chars = '京津冀晋蒙辽吉黑沪苏浙皖闽赣鲁豫鄂湘粤桂琼渝川贵云藏陕甘青宁新#ABCDEFGHJKLMNPQRSTUVWXYZ0123456789警港澳学领使挂'
    converter = CTCLabelConverter(all_chars)
    dataset = TestLmdbDataset("/home/dl/liyunfei/project/rec_lmdb_dataset/test_db", "cennavi_v1", sensitive=False)
    collate_fn = TestAlignCollate(0.5, 0.5, args.min_w, args.input_c, args.input_h, args.input_w)
    dataloader = DataLoader(dataset, batch_size=32, collate_fn=collate_fn)

    res = []
    with torch.no_grad():
        for i, (indexes, input_batch, labels) in enumerate(dataloader):
            batch_size = len(indexes)
            if batch_size == 0: continue

            pred_probability = model(input_batch.cuda()).softmax(2)
            pred_probability_max, pred_index = pred_probability.max(2)         # pred_index: (batch_size, seq_length)
            pred_index = pred_index.contiguous().view(-1)
            pred_seq_length = pred_probability.size(1)
            pred_text = converter.decode(pred_index.data, [pred_seq_length] * batch_size)
            res.extend(list(zip(indexes, labels, pred_text, pred_probability_max.mean(axis=1).data.tolist())))

    char_acc, line_acc = [], []
    for idx, label, pred, score in res:
        if "#" in label:
            continue
        if score > args.score_t:
            char_acc.append(get_char_acc(pred[1:], label[1:], "#"))
            line_acc.append(get_line_acc(pred[1:], label[1:], "#"))
    print(len(char_acc))
    print(sum(char_acc)/len(char_acc), sum(line_acc)/len(line_acc))
    print(res[:100])


if __name__ == "__main__":
    main()
