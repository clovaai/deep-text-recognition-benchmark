# What Is Wrong With Scene Text Recognition Model Comparisons? Dataset and Model Analysis
| [paper](https://arxiv.org/abs/1904.01906) | [training and evaluation data](https://github.com/clovaai/deep-text-recognition-benchmark#download-lmdb-dataset-for-traininig-and-evaluation-from-here) | [failure cases and cleansed label](https://github.com/clovaai/deep-text-recognition-benchmark#download-failure-cases-and-cleansed-label-from-here) | [pretrained model](https://drive.google.com/drive/folders/15WPsuPJDCzhp2SvYZLRj8mAlT3zmoAMW) | [Baidu ver(passwd:rryk)](https://pan.baidu.com/s/1KSNLv4EY3zFWHpBYlpFCBQ) |

Official PyTorch implementation of our four-stage STR framework, that most existing STR models fit into. <br>
Using this framework allows for the module-wise contributions to performance in terms of accuracy, speed, and memory demand, under one consistent set of training and evaluation datasets. <br>
Such analyses clean up the hindrance on the current comparisons to understand the performance gain of the existing modules. <br><br>
<img src="./figures/trade-off.png" width="1000" title="trade-off">

## Honors
Based on this framework, we recorded the 1st place of [ICDAR2013 focused scene text](https://rrc.cvc.uab.es/?ch=2&com=evaluation&task=3), [ICDAR2019 ArT](https://rrc.cvc.uab.es/files/ICDAR2019-ArT.pdf) and 3rd place of [ICDAR2017 COCO-Text](https://rrc.cvc.uab.es/?ch=5&com=evaluation&task=2), [ICDAR2019 ReCTS (task1)](https://rrc.cvc.uab.es/files/ICDAR2019-ReCTS.pdf). <br>
The difference between our paper and ICDAR challenge is summarized [here](https://github.com/clovaai/deep-text-recognition-benchmark/issues/13).

## Updates
**Aug 3, 2020**: added [guideline to use Baidu warpctc](https://github.com/clovaai/deep-text-recognition-benchmark/pull/209) which reproduces CTC results of our paper. <br>
**Dec 27, 2019**: added [FLOPS](https://github.com/clovaai/deep-text-recognition-benchmark/issues/125) in our paper, and minor updates such as log_dataset.txt and [ICDAR2019-NormalizedED](https://github.com/clovaai/deep-text-recognition-benchmark/blob/86451088248e0490ff8b5f74d33f7d014f6c249a/test.py#L139-L165). <br>
**Oct 22, 2019**: added [confidence score](https://github.com/clovaai/deep-text-recognition-benchmark/issues/82), and arranged the output form of training logs. <br>
**Jul 31, 2019**: The paper is accepted at International Conference on Computer Vision (ICCV), Seoul 2019, as an oral talk. <br>
**Jul 25, 2019**: The code for floating-point 16 calculation, check [@YacobBY's](https://github.com/YacobBY) [pull request](https://github.com/clovaai/deep-text-recognition-benchmark/pull/36) <br>
**Jul 16, 2019**: added [ST_spe.zip](https://drive.google.com/drive/folders/192UfE9agQUMNq6AgU3_E05_FcPZK4hyt) dataset, word images contain special characters in SynthText (ST) dataset, see [this issue](https://github.com/clovaai/deep-text-recognition-benchmark/issues/7#issuecomment-511727025) <br>
**Jun 24, 2019**: added gt.txt of failure cases that contains path and label of each image, see [image_release_190624.zip](https://drive.google.com/open?id=1VAP9l5GL5fgptgKDLio_h3nMe7X9W0Mf) <br>
**May 17, 2019**: uploaded resources in Baidu Netdisk also, added [Run demo](https://github.com/clovaai/deep-text-recognition-benchmark#run-demo-with-pretrained-model). (check [@sharavsambuu's](https://github.com/sharavsambuu) [colab demo also](https://colab.research.google.com/drive/1PHnc_QYyf9b1_KJ1r15wYXaOXkdm1Mrk)) <br>
**May 9, 2019**: PyTorch version updated from 1.0.1 to 1.1.0, use torch.nn.CTCLoss instead of torch-baidu-ctc, and various minor updated.

## Getting Started
### Dependency
- This work was tested with PyTorch 1.3.1, CUDA 10.1, python 3.6 and Ubuntu 16.04. <br> You may need `pip3 install torch==1.3.1`. <br>
In the paper, expriments were performed with **PyTorch 0.4.1, CUDA 9.0**.
- requirements : lmdb, pillow, torchvision, nltk, natsort
```
pip3 install lmdb pillow torchvision nltk natsort
```

### Download lmdb dataset for traininig and evaluation from [here](https://drive.google.com/drive/folders/192UfE9agQUMNq6AgU3_E05_FcPZK4hyt)
data_lmdb_release.zip contains below. <br>
training datasets : [MJSynth (MJ)](http://www.robots.ox.ac.uk/~vgg/data/text/)[1] and [SynthText (ST)](http://www.robots.ox.ac.uk/~vgg/data/scenetext/)[2] \
validation datasets : the union of the training sets [IC13](http://rrc.cvc.uab.es/?ch=2)[3], [IC15](http://rrc.cvc.uab.es/?ch=4)[4], [IIIT](http://cvit.iiit.ac.in/projects/SceneTextUnderstanding/IIIT5K.html)[5], and [SVT](http://www.iapr-tc11.org/mediawiki/index.php/The_Street_View_Text_Dataset)[6].\
evaluation datasets : benchmark evaluation datasets, consist of [IIIT](http://cvit.iiit.ac.in/projects/SceneTextUnderstanding/IIIT5K.html)[5], [SVT](http://www.iapr-tc11.org/mediawiki/index.php/The_Street_View_Text_Dataset)[6], [IC03](http://www.iapr-tc11.org/mediawiki/index.php/ICDAR_2003_Robust_Reading_Competitions)[7], [IC13](http://rrc.cvc.uab.es/?ch=2)[3], [IC15](http://rrc.cvc.uab.es/?ch=4)[4], [SVTP](http://openaccess.thecvf.com/content_iccv_2013/papers/Phan_Recognizing_Text_with_2013_ICCV_paper.pdf)[8], and [CUTE](http://cs-chan.com/downloads_CUTE80_dataset.html)[9].

### Run demo with pretrained model
1. Download pretrained model from [here](https://drive.google.com/drive/folders/15WPsuPJDCzhp2SvYZLRj8mAlT3zmoAMW)
2. Add image files to test into `demo_image/`
3. Run demo.py (add `--sensitive` option if you use case-sensitive model)
```
CUDA_VISIBLE_DEVICES=0 python3 demo.py \
--Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction Attn \
--image_folder demo_image/ \
--saved_model TPS-ResNet-BiLSTM-Attn.pth
```

#### prediction results

| demo images | [TRBA (**T**PS-**R**esNet-**B**iLSTM-**A**ttn)](https://drive.google.com/open?id=1b59rXuGGmKne1AuHnkgDzoYgKeETNMv9) | [TRBA (case-sensitive version)](https://drive.google.com/open?id=1ajONZOgiG9pEYsQ-eBmgkVbMDuHgPCaY) |
| ---         |     ---      |          --- |
| <img src="./demo_image/demo_1.png" width="300">    |   available   |  Available   |
| <img src="./demo_image/demo_2.jpg" width="300">      |    shakeshack    |   SHARESHACK    |
| <img src="./demo_image/demo_3.png" width="300">  |   london   |  Londen   |
| <img src="./demo_image/demo_4.png" width="300">      |    greenstead    |   Greenstead    |
| <img src="./demo_image/demo_5.png" width="300" height="100">    |   toast   |  TOAST   |
| <img src="./demo_image/demo_6.png" width="300" height="100">      |    merry    |   MERRY    |
| <img src="./demo_image/demo_7.png" width="300">    |   underground   |   underground  |
| <img src="./demo_image/demo_8.jpg" width="300">      |    ronaldo    |    RONALDO   |
| <img src="./demo_image/demo_9.jpg" width="300" height="100">    |   bally   |   BALLY  |
| <img src="./demo_image/demo_10.jpg" width="300" height="100">      |    university    |   UNIVERSITY    |


### Training and evaluation
1. Train CRNN[10] model
```
CUDA_VISIBLE_DEVICES=0 python3 train.py \
--train_data data_lmdb_release/training --valid_data data_lmdb_release/validation \
--select_data MJ-ST --batch_ratio 0.5-0.5 \
--Transformation None --FeatureExtraction VGG --SequenceModeling BiLSTM --Prediction CTC
```
2. Test CRNN[10] model. If you want to evaluate IC15-2077, check [data filtering part](https://github.com/clovaai/deep-text-recognition-benchmark/blob/c27abe6b4c681e2ee0784ad966602c056a0dd3b5/dataset.py#L148). 
```
CUDA_VISIBLE_DEVICES=0 python3 test.py \
--eval_data data_lmdb_release/evaluation --benchmark_all_eval \
--Transformation None --FeatureExtraction VGG --SequenceModeling BiLSTM --Prediction CTC \
--saved_model saved_models/None-VGG-BiLSTM-CTC-Seed1111/best_accuracy.pth
```

3. Try to train and test our best accuracy model TRBA (**T**PS-**R**esNet-**B**iLSTM-**A**ttn) also. ([download pretrained model](https://drive.google.com/drive/folders/15WPsuPJDCzhp2SvYZLRj8mAlT3zmoAMW))
```
CUDA_VISIBLE_DEVICES=0 python3 train.py \
--train_data data_lmdb_release/training --valid_data data_lmdb_release/validation \
--select_data MJ-ST --batch_ratio 0.5-0.5 \
--Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction Attn
```
```
CUDA_VISIBLE_DEVICES=0 python3 test.py \
--eval_data data_lmdb_release/evaluation --benchmark_all_eval \
--Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction Attn \
--saved_model saved_models/TPS-ResNet-BiLSTM-Attn-Seed1111/best_accuracy.pth
```

### Arguments
* `--train_data`: folder path to training lmdb dataset.
* `--valid_data`: folder path to validation lmdb dataset.
* `--eval_data`: folder path to evaluation (with test.py) lmdb dataset.
* `--select_data`: select training data. default is MJ-ST, which means MJ and ST used as training data.
* `--batch_ratio`: assign ratio for each selected data in the batch. default is 0.5-0.5, which means 50% of the batch is filled with MJ and the other 50% of the batch is filled ST.
* `--data_filtering_off`: skip [data filtering](https://github.com/clovaai/deep-text-recognition-benchmark/blob/f2c54ae2a4cc787a0f5859e9fdd0e399812c76a3/dataset.py#L126-L146) when creating LmdbDataset. 
* `--Transformation`: select Transformation module [None | TPS].
* `--FeatureExtraction`: select FeatureExtraction module [VGG | RCNN | ResNet].
* `--SequenceModeling`: select SequenceModeling module [None | BiLSTM].
* `--Prediction`: select Prediction module [CTC | Attn].
* `--saved_model`: assign saved model to evaluation.
* `--benchmark_all_eval`: evaluate with 10 evaluation dataset versions, same with Table 1 in our paper.

## Download failure cases and cleansed label from [here](https://drive.google.com/drive/folders/1W84gS9T5GU5l5Wp3VV1aeXIIKV87yjRm)
image_release.zip contains failure case images and benchmark evaluation images with cleansed label.
<img src="./figures/failure-case.jpg" width="1000" title="failure cases">

## When you need to train on your own dataset or Non-Latin language datasets.
1. Create your own lmdb dataset.
```
pip3 install fire
python3 create_lmdb_dataset.py --inputPath data/ --gtFile data/gt.txt --outputPath result/
```
The structure of data folder as below.
```
data
├── gt.txt
└── test
    ├── word_1.png
    ├── word_2.png
    ├── word_3.png
    └── ...
```
At this time, `gt.txt` should be `{imagepath}\t{label}\n` <br>
For example
```
test/word_1.png Tiredness
test/word_2.png kills
test/word_3.png A
...
```
2. Modify `--select_data`, `--batch_ratio`, and `opt.character`, see [this issue](https://github.com/clovaai/deep-text-recognition-benchmark/issues/85).


## Acknowledgements
This implementation has been based on these repository [crnn.pytorch](https://github.com/meijieru/crnn.pytorch), [ocr_attention](https://github.com/marvis/ocr_attention).

## Reference
[1] M. Jaderberg, K. Simonyan, A. Vedaldi, and A. Zisserman. Synthetic data and artificial neural networks for natural scenetext  recognition. In Workshop on Deep Learning, NIPS, 2014. <br>
[2] A. Gupta, A. Vedaldi, and A. Zisserman. Synthetic data fortext localisation in natural images. In CVPR, 2016. <br>
[3] D. Karatzas, F. Shafait, S. Uchida, M. Iwamura, L. G. i Big-orda, S. R. Mestre, J. Mas, D. F. Mota, J. A. Almazan, andL. P. De Las Heras. ICDAR 2013 robust reading competition. In ICDAR, pages 1484–1493, 2013. <br>
[4] D. Karatzas, L. Gomez-Bigorda, A. Nicolaou, S. Ghosh, A. Bagdanov, M. Iwamura, J. Matas, L. Neumann, V. R.Chandrasekhar, S. Lu, et al. ICDAR 2015 competition on ro-bust reading. In ICDAR, pages 1156–1160, 2015. <br>
[5] A. Mishra, K. Alahari, and C. Jawahar. Scene text recognition using higher order language priors. In BMVC, 2012. <br>
[6] K. Wang, B. Babenko, and S. Belongie. End-to-end scenetext recognition. In ICCV, pages 1457–1464, 2011. <br>
[7] S. M. Lucas, A. Panaretos, L. Sosa, A. Tang, S. Wong, andR. Young. ICDAR 2003 robust reading competitions. In ICDAR, pages 682–687, 2003. <br>
[8] T. Q. Phan, P. Shivakumara, S. Tian, and C. L. Tan. Recognizing text with perspective distortion in natural scenes. In ICCV, pages 569–576, 2013. <br>
[9] A. Risnumawan, P. Shivakumara, C. S. Chan, and C. L. Tan. A robust arbitrary text detection system for natural scene images. In ESWA, volume 41, pages 8027–8048, 2014. <br>
[10] B. Shi, X. Bai, and C. Yao. An end-to-end trainable neural network for image-based sequence recognition and its application to scene text recognition. In TPAMI, volume 39, pages2298–2304. 2017. 

## Links
- WebDemo : https://demo.ocr.clova.ai/ <br>
Combination of Clova AI detection and recognition, additional/advanced features used for KOR/JPN.
- Repo of detection : https://github.com/clovaai/CRAFT-pytorch

## Citation
Please consider citing this work in your publications if it helps your research.
```
@inproceedings{baek2019STRcomparisons,
  title={What Is Wrong With Scene Text Recognition Model Comparisons? Dataset and Model Analysis},
  author={Baek, Jeonghun and Kim, Geewook and Lee, Junyeop and Park, Sungrae and Han, Dongyoon and Yun, Sangdoo and Oh, Seong Joon and Lee, Hwalsuk},
  booktitle = {International Conference on Computer Vision (ICCV)},
  year={2019},
  pubstate={published},
  tppubtype={inproceedings}
}
```

## Contact
Feel free to contact us if there is any question: <br>
for code/paper Jeonghun Baek ku21fang@gmail.com; for collaboration hwalsuk.lee@navercorp.com (our team leader).

## License
Copyright (c) 2019-present NAVER Corp.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

