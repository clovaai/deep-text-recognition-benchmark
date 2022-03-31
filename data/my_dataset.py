import random
import re

import cv2
import lmdb
import numpy as np
import torch
import torchvision.transforms.transforms as T
from PIL import Image
from torch.utils.data import Dataset

from .augmentation import random_blur, random_flat


class TestLmdbDataset(Dataset):

    def __init__(self, root, db_name, sensitive):
        super(TestLmdbDataset, self).__init__()
        self.sensitive = sensitive
        self.env = lmdb.open(root, max_dbs=1, max_readers=32, readonly=True,
                             lock=False, readahead=False, meminit=False)
        self.db = self.env.open_db(db_name.encode("utf8"))
        if not self.env:
            raise IOError(f"cannot create lmdb from {root}")
        with self.env.begin(write=False) as txn:
            self.num_samples = int(txn.get('num-samples'.encode("utf8"), db=self.db).decode("utf8"))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):

        with self.env.begin(write=False) as txn:
            # text label
            label_key = 'label-%d'.encode('utf-8') % index
            label = txn.get(label_key, db=self.db).decode('utf-8')
            if not self.sensitive:
                label = label.upper()

            # image
            img_key = 'image-%d'.encode('utf-8') % index
            np_byte_encode = txn.get(img_key, db=self.db)
            np_arr_decode = np.frombuffer(np_byte_encode, np.uint8)
            img = cv2.imdecode(np_arr_decode, cv2.IMREAD_COLOR)

        return index, img, label


class TestAlignCollate(object):

    def __init__(self, mean, std, min_w=0, input_c=3, input_h=32, input_w=None):
        mean = [mean] if isinstance(mean, (float, int)) else mean
        std = [std] if isinstance(std, (float, int)) else std
        assert max(mean) < 1., max(std) < 1.
        self.mean = 255. * torch.tensor(mean)[None, :, None, None]
        self.std = 255. * torch.tensor(std)[None, :, None, None]
        self.min_w = min_w
        self.input_c = input_c
        self.input_h = input_h
        self.input_w = input_w

    def __call__(self, batch_list):
        indexes, images, labels = zip(*batch_list)
        indexes, images, labels = self._filter_sample_via_min_width(indexes, images, labels)
        batch_size = len(images)
        if batch_size == 0:
            return [], [], []

        input_w = max(it.shape[1] for it in images) if not self.input_w else self.input_w
        input_tensor = torch.zeros((batch_size, self.input_c, self.input_h, input_w))
        for i, image in enumerate(images):
            # color
            if self.input_c == 1:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # resize
            h, w = image.shape[:2]
            new_h = self.input_h
            new_w = min(int((new_h / h) * w), self.input_w)
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

            input_tensor[i, :, :, :new_w] = torch.from_numpy(image).permute(2, 0, 1).contiguous().float()
        input_tensor.sub_(self.mean).div_(self.std)

        return indexes, input_tensor, labels

    def _filter_sample_via_min_width(self, indexes, images, labels):
        res_indexes, res_images, res_labels = [], [], []
        for idx, image, label in zip(indexes, images, labels):
            if image.shape[1] >= self.min_w:
                res_indexes.append(idx)
                res_images.append(image)
                res_labels.append(label)
        return res_indexes, res_images, res_labels


class MyLmdbDataset(Dataset):

    def __init__(self, root, db_name, max_length, all_characters, input_w, input_h, input_c, mean, std, sensitive,
                 do_trans: bool):
        """ lmdb 数据集，生成 rec 用的数据

        Args:
            root (str): lmdb 数据集路径
            db_name (str): 使用的数据集名称
            max_length (int): label 的最大长度, 长度超出
            all_characters (str): 支持的字符集
            input_w, input_h, input_c (int): 输入分辨率
            mean, std (float): 均值和方差(0-1之间)
            sensitive (bool): 英文大小写是否敏感, 如果否, 全部使用 upper
            do_trans (bool): 是否做数据增强(训练集需要做)
        """

        self.root = root
        self.db_name = db_name
        self.max_length = max_length
        self.all_characters = all_characters
        self.mean, self.std = mean, std,
        self.input_c, self.input_w, self.input_h = input_c, input_w, input_h
        self.sensitive = sensitive
        self.do_trans = do_trans

        self.env = lmdb.open(root, max_dbs=1, max_readers=32, readonly=True,
                             lock=False, readahead=False, meminit=False)
        self.db = self.env.open_db(db_name.encode("utf8"))
        if not self.env:
            raise IOError(f"cannot create lmdb from {root}")

        with self.env.begin(write=False) as txn:
            num_samples = int(txn.get('num-samples'.encode("utf8"), db=self.db).decode("utf8"))

            # Filtering
            self.filtered_index_list = []
            for index in range(num_samples):
                label_key = 'label-%d'.encode('utf-8') % index
                label = txn.get(label_key, db=self.db).decode('utf-8')

                if len(label) > self.max_length:
                    print(f'The length of the label is longer than max_length: {max_length} \
                        {len(label)}, {label} in dataset {self.root}: {self.db_name}: {label_key}')
                    continue

                # By default, images containing characters which are not in opt.character are filtered.
                # You can add [UNK] token to `opt.character` in utils.py instead of this filtering.
                out_of_char = f'[^{self.all_characters}]'
                if re.search(out_of_char, label):
                    # print(
                    #     f'There are invalided characters in: {label} in dataset {self.root}: {self.db_name}: {label_key}')
                    continue

                self.filtered_index_list.append(index)

            self.num_samples = len(self.filtered_index_list)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        index = self.filtered_index_list[index]

        with self.env.begin(write=False) as txn:
            # text label
            label_key = 'label-%d'.encode('utf-8') % index
            label = txn.get(label_key, db=self.db).decode('utf-8')
            if not self.sensitive:
                label = label.upper()
            # image
            img_key = 'image-%d'.encode('utf-8') % index
            np_byte_encode = txn.get(img_key, db=self.db)
            np_arr_decode = np.frombuffer(np_byte_encode, np.uint8)
            img = cv2.imdecode(np_arr_decode, cv2.IMREAD_COLOR)

            if self.input_c == 1:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # data augmentation
            if self.do_trans:
                img = Image.fromarray(img)
                # 数据增强
                if random.random() < 0.1:  # 0.2:
                    img = random_flat(img, 0.7, 1.33)
                if random.random() < 0.2:
                    img = random_blur(img, random.randint(1, 3))
                if random.random() < 0.7:  # 0.3:
                    img = T.ColorJitter(brightness=0.4, contrast=0.4)(img)
                img = np.asarray(img)

            # resize
            h, w = img.shape[:2]
            h_ = self.input_h
            w_ = int((h_ / h) * w)
            if w_ < self.input_w:
                # padding
                img = cv2.resize(img, (w_, h_), interpolation=cv2.INTER_CUBIC)
                if self.input_c == 3:
                    pad = ((0, self.input_h - h_), (0, self.input_w - w_), (0, 0))
                else:
                    pad = ((0, self.input_h - h_), (0, self.input_w - w_))
                img = np.pad(img, pad, 'constant', constant_values=0)
            else:
                # resize
                img = cv2.resize(img, (self.input_w, self.input_h), interpolation=cv2.INTER_CUBIC)

            # normlization
            if self.input_c == 3:
                img = torch.from_numpy(img).permute(2, 0, 1).contiguous().float()
            else:
                img = torch.from_numpy(img[None]).contiguous().float()
            img.div_(255.).sub_(self.mean).div_(self.std)

        return img, label


if __name__ == "__main__":
    config = {
        "input_channels": 3,  # RGB
        "input_height": 32,
        "input_width": 100,
        # "sensitive": False, # 是否大小写敏感?
        "batch_max_length": 10,  # 标签最大长度
        "character": "#皖沪津渝冀晋蒙辽吉黑苏浙京闽赣鲁豫鄂湘粤桂琼川贵云西陕甘青宁新abcdefghjklmnpqrstuvwxyz0123456789",  # 合理的字符集, # 表示 unknow word
    }
    provinces = ["京", "津", "冀", "晋", "蒙", "辽", "吉", "黑", "沪",
                 "苏", "浙", "皖", "闽", "赣", "鲁", "豫", "鄂", "湘",
                 "粤", "桂", "琼", "渝", "川", "贵", "云", "藏", "陕",
                 "甘", "青", "宁", "新"]
    all_characters = "".join(provinces) + "#abcdefghjklmnpqrstuvwxyz0123456789警港澳学领使挂"
    dataset = MyLmdbDataset(
        "/home/dl/liyunfei/project/rec_lmdb_dataset/train_val_db",
        "cennavi_v1", 10, all_characters, 100, 32, 3, 0.5, 0.5, True)

    from torch.utils.data import DataLoader

    dataloader = DataLoader(
        dataset, batch_size=32
    )
    for img, label in dataloader:
        print(img.shape, label)
        break
