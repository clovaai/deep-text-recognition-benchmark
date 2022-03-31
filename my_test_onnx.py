import argparse
import onnx
import onnxruntime as ort
import numpy as np
import torch
from torch.utils.data import DataLoader
from data.my_dataset import TestLmdbDataset, TestAlignCollate
from utils import CTCLabelConverter, get_char_acc, get_line_acc


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx_path", default="", help="path to onnx checkpoint.")
    parser.add_argument("--min_w", type=float, default=0, help="min width of test image.")
    parser.add_argument("--score_t", type=float, default=0.9, help="filter threshold of recognize result.")
    parser.add_argument("--input_c", type=int, default=3)
    parser.add_argument("--input_h", type=int, default=32)
    parser.add_argument("--input_w", type=int, default=160)

    return parser.parse_args()

def check_onnx(onnx_path):
    # Load the ONNX model
    model = onnx.load(onnx_path)
    # Check that the model is well formed
    onnx.checker.check_model(model)
    # Print a human readable representation of the graph
    print(onnx.helper.printable_graph(model.graph))
    

def main():
    args = get_args()
    args.onnx_path = "/home/dl/liyunfei/project/lp_dev_v1/licenseplate_recognition/recog_dev/output/release/lp_rec_0.1.0.onnx"
    args.input_c = 3
    args.input_h = 64
    args.input_w = 200

    ort_session = ort.InferenceSession(
        args.onnx_path,
        providers=['CUDAExecutionProvider']
    )

    all_chars = '京津冀晋蒙辽吉黑沪苏浙皖闽赣鲁豫鄂湘粤桂琼渝川贵云藏陕甘青宁新#ABCDEFGHJKLMNPQRSTUVWXYZ0123456789警港澳学领使挂'
    all_chars = "".join(["0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
                       "皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏",
                       "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂", "琼",
                       "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "警", "学",
                       "澳", "港", "领", "使", "挂", "A", "B", "C", "D", "E", "F", "G",
                       "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
                       "U", "V", "W", "X", "Y", "Z"])

    converter = CTCLabelConverter(all_chars)
    dataset = TestLmdbDataset("/home/dl/liyunfei/project/rec_lmdb_dataset/test_db", "cennavi_v1", sensitive=False)
    collate_fn = TestAlignCollate(0.5, 0.5, args.min_w, args.input_c, args.input_h, args.input_w)
    dataloader = DataLoader(dataset, batch_size=32, collate_fn=collate_fn)
    res = []
    for i, (indexes, input_batch, labels) in enumerate(dataloader):
        batch_size = len(indexes)
        if batch_size == 0: continue

        pred_probability = ort_session.run(None, {"input": input_batch.numpy().astype(np.float32)})[0]
        pred_probability = torch.from_numpy(pred_probability)       # TODO 全部用 numpy
        pred_probability_max, pred_index = pred_probability.max(2)  # pred_index: (batch_size, seq_length)
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
    print(sum(char_acc) / len(char_acc), sum(line_acc) / len(line_acc))
    print(res[:100])

    
if __name__ == "__main__":
    print(ort.get_device())
    onnx_path = "/home/dl/liyunfei/project/lp_dev_v1/licenseplate_recognition/recog_dev/output/release/lp_rec_0.1.0.onnx"
    # check_onnx(onnx_path)
    main()
