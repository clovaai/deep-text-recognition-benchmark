""" a modified version of CRNN torch repository https://github.com/bgshih/crnn/blob/master/tool/create_dataset.py """

import fire
import os
import lmdb
import cv2
import os
import pandas as pd
import numpy as np


def generate_values_file(parent_folder):
    """
    Create LMDB dataset for training and evaluation.
    ARGS:
        inputPath  : input folder path where starts imagePath
        outputPath : LMDB output path
        gtFile     : list of image path and label
        checkValid : if true, check the validity of every image
    """

    #parent_folder = "/home/raki-dedigama/projects/ocr-ean/ocr-datasets/labelled-data/kar-smm/"
    folders = [folder for folder in os.listdir(parent_folder)]
    for folder in folders :

        image_folder_path = parent_folder + folder
        print(image_folder_path)
        #annotations_df = pd.DataFrame(columns = ['file_name', 'text'])
        files = [file for file in os.listdir(image_folder_path) if os.path.isdir(image_folder_path)]
        image_label_list = []
        for file_name in files:             
            if file_name.endswith('.jpg') or file_name.endswith('.png'):
                image_label = str.split(file_name, '_')[0]
                dict = {}
                dict = {'file_name': file_name,
                    'text' : image_label
                }
                image_label_list.append(dict)
        annotations_df = pd.DataFrame(image_label_list, columns = ['file_name', 'text'])
        annotations_df.to_csv(image_folder_path + "/values.csv", index = None)
                #print("image : ", file_name, "\t label : ", image_label)        
                
    #print(files)
    
    #print('Created dataset with %d samples' % nSamples)



if __name__ == '__main__':
    fire.Fire(generate_values_file)



