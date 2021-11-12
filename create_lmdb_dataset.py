""" a modified version of CRNN torch repository https://github.com/bgshih/crnn/blob/master/tool/create_dataset.py """

import fire
import os
import lmdb
import cv2
import os
import pandas as pd
import numpy as np


def checkImageIsValid(imageBin):
    if imageBin is None:
        return False
    imageBuf = np.frombuffer(imageBin, dtype=np.uint8)
    img = cv2.imdecode(imageBuf, cv2.IMREAD_GRAYSCALE)
    imgH, imgW = img.shape[0], img.shape[1]
    if imgH * imgW == 0:
        return False
    return True


def writeCache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            txn.put(k, v)


def createDataset(inputPath, gtFile, outputPath, checkValid=True):
    """
    Create LMDB dataset for training and evaluation.
    ARGS:
        inputPath  : input folder path where starts imagePath
        outputPath : LMDB output path
        gtFile     : list of image path and label
        checkValid : if true, check the validity of every image
    """
    os.makedirs(outputPath, exist_ok=True)
    env = lmdb.open(outputPath, map_size=1099511627776)
    cache = {}
    cnt = 1

    with open(gtFile, 'r', encoding='utf-8') as data:
        datalist = data.readlines()

    nSamples = len(datalist)
    for i in range(nSamples):
        imagePath, label = datalist[i].strip('\n').split('\t')
        imagePath = os.path.join(inputPath, imagePath)

        # # only use alphanumeric data
        # if re.search('[^a-zA-Z0-9]', label):
        #     continue

        if not os.path.exists(imagePath):
            print('%s does not exist' % imagePath)
            continue
        with open(imagePath, 'rb') as f:
            imageBin = f.read()
        if checkValid:
            try:
                if not checkImageIsValid(imageBin):
                    print('%s is not a valid image' % imagePath)
                    continue
            except:
                print('error occured', i)
                with open(outputPath + '/error_image_log.txt', 'a') as log:
                    log.write('%s-th image data occured error\n' % str(i))
                continue

        imageKey = 'image-%09d'.encode() % cnt
        labelKey = 'label-%09d'.encode() % cnt
        cache[imageKey] = imageBin
        cache[labelKey] = label.encode()

        if cnt % 1000 == 0:
            writeCache(env, cache)
            cache = {}
            print('Written %d / %d' % (cnt, nSamples))
        cnt += 1
    nSamples = cnt-1
    cache['num-samples'.encode()] = str(nSamples).encode()
    writeCache(env, cache)
    print('Created dataset with %d samples' % nSamples)



def generate_gt_file_for_dataset(data_path):

    gt_file = data_path + 'gt.txt'
    print(gt_file)
    annotations_df = pd.DataFrame(columns = ['file_name', 'text'])
    folders = [folder for folder in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, folder))]
    for folder in folders:
        print(folder)
        gt_folder_path = data_path + folder + '/' + 'values.csv'
        if os.path.isfile(gt_folder_path):
            gt_folder_df   = pd.read_csv(gt_folder_path)

            ## remove entries with non-digit labels (illegible images)
            gt_folder_df = gt_folder_df.loc[gt_folder_df['text']!= '_']
            ## append folder path to image paths
            gt_folder_df['file_name'] = folder + '/' + gt_folder_df['file_name']
            # concatenate folder annotatoins to train annotations
            annotations_df = pd.concat([annotations_df, gt_folder_df])
        else : 
            print("Could not find values.csv in folder ", folder, ". Skip folder")

    print(annotations_df.shape)
    print(annotations_df.head())

    annotations_df.to_csv(gt_file, header=None, index = None, sep = '\t', mode='a') 


def create_lmdb_dataset(datasetPath,inputPath, gtFile, outputPath ):
    generate_gt_file_for_dataset(datasetPath )
    createDataset(inputPath, gtFile, outputPath, checkValid=True)

if __name__ == '__main__':
    fire.Fire(create_lmdb_dataset)



