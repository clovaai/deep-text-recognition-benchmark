from dataset import LmdbDataset
import argparse
import string

def read_mdb(opt):
    nick = 'result/nickname'
    nick_val = 'result/nickname_val'

    dataset = LmdbDataset(nick_val,opt)

    for idx, data in enumerate(dataset):
        print(data)
        if idx==10:
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
    parser.add_argument('--batch_size', type=int, default=192, help='input batch size')
    """ Data processing """
    parser.add_argument('--batch_max_length', type=int, default=45, help='maximum-label-length')
    parser.add_argument('--imgH', type=int, default=32, help='the height of the input image')
    parser.add_argument('--imgW', type=int, default=100, help='the width of the input image')
    parser.add_argument('--rgb', action='store_true', help='use rgb input')
    parser.add_argument('--character', type=str, default='0123456789abcdefghijklmnopqrstuvwxyz', help='character label')
    parser.add_argument('--PAD', action='store_true', help='whether to keep ratio then pad for image resize')
    """ Model Architecture """
    parser.add_argument('--num_fiducial', type=int, default=20, help='number of fiducial points of TPS-STN')
    parser.add_argument('--input_channel', type=int, default=1, help='the number of input channel of Feature extractor')
    parser.add_argument('--output_channel', type=int, default=512,
                        help='the number of output channel of Feature extractor')
    parser.add_argument('--hidden_size', type=int, default=256, help='the size of the LSTM hidden state')
    parser.add_argument('--data_filtering_off', action='store_true', help='for data_filtering_off mode')
    parser.add_argument('--sensitive', action='store_true', help='for sensitive character mode')

    opt = parser.parse_args()

    charlist = []
    with open('ko_char.txt', 'r', encoding='utf-8') as f:
        for c in f.readlines():
            charlist.append(c[:-1])
    opt.character = ''.join(charlist) + string.printable[:-38]

    opt.data_filtering_off = True

    read_mdb(opt)