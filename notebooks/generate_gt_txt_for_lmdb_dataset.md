```python
dataset_path = '../../ocr-datasets/kar-lmdb/'
train_data_path = dataset_path + 'kar-train/'
valid_data_path = dataset_path + 'kar-valid/'
test_data_path  = dataset_path + 'kar-test/'

```


```python
## train data

import os
import pandas as pd

train_annotations_df = pd.DataFrame(columns = ['file_name', 'text'])
train_folders = [folder for folder in os.listdir(train_data_path) if os.path.isdir(os.path.join(train_data_path, folder))]

for folder in train_folders:
    print(folder)
    gt_folder_path = train_data_path + folder + '/' + 'values.csv'
    gt_folder_df   = pd.read_csv(gt_folder_path)
    
    ## remove entries with non-digit labels (illegible images)
    gt_folder_df = gt_folder_df.loc[gt_folder_df['text']!= '_']
  
    
    ## append folder path to image paths
    gt_folder_df['file_name'] = folder + '/' + gt_folder_df['file_name']

    
    # concatenate folder annotatoins to train annotations
    train_annotations_df = pd.concat([train_annotations_df, gt_folder_df])
    

print(train_annotations_df.shape)
print(train_annotations_df.head())

gt_train_path = train_data_path + 'gt-train.txt'
train_annotations_df.to_csv(gt_train_path, header=None, index = None, sep = '\t', mode='a')
```

    PRODUCT_NUMBER.3.annotated
    PRODUCT_NUMBER.0.annotated
    PRODUCT_NUMBER.1.annotated
    PRODUCT_NUMBER.2.annotated
    (764, 2)
                                               file_name           text
    0  PRODUCT_NUMBER.3.annotated/3.0.P0_1C14D905.160...  6412203102230
    1  PRODUCT_NUMBER.3.annotated/4.0.P1_1615C405.160...  5412386013829
    2  PRODUCT_NUMBER.3.annotated/4.0.P1_1615C405.160...  8011017793009
    3  PRODUCT_NUMBER.3.annotated/8.0.P2_1B15D905.160...  7046110037393
    4  PRODUCT_NUMBER.3.annotated/10.0.P1_1615C405.16...  5412386063503



```python
## valid data

import os
import pandas as pd

valid_annotations_df = pd.DataFrame(columns = ['file_name', 'text'])
valid_folders = [folder for folder in os.listdir(valid_data_path) if os.path.isdir(os.path.join(valid_data_path, folder))]

for folder in valid_folders:
    print(folder)
    gt_folder_path = valid_data_path + folder + '/' + 'values.csv'
    gt_folder_df   = pd.read_csv(gt_folder_path)
    
    ## remove entries with non-digit labels (illegible images)
    gt_folder_df = gt_folder_df.loc[gt_folder_df['text']!= '_']
  
    
    ## append folder path to image paths
    gt_folder_df['file_name'] = folder + '/' + gt_folder_df['file_name']

    
    # concatenate folder annotatoins to train annotations
    valid_annotations_df = pd.concat([valid_annotations_df, gt_folder_df])
    

print(valid_annotations_df.shape)
print(valid_annotations_df.head())

gt_valid_path = valid_data_path + 'gt-valid.txt'
valid_annotations_df.to_csv(gt_valid_path, header=None, index = None, sep = '\t', mode='a')
```

    PRODUCT_NUMBER.4.annotated
    (192, 2)
                                               file_name           text
    0  PRODUCT_NUMBER.4.annotated/1.0.P1_1615C405.160...  6438168092263
    1  PRODUCT_NUMBER.4.annotated/2.0.P1_1615C405.160...  6414671011144
    2  PRODUCT_NUMBER.4.annotated/6.0.P2_1B15D905.160...  9312657011010
    3  PRODUCT_NUMBER.4.annotated/6.0.P2_1B15D905.160...  9312657007013
    4  PRODUCT_NUMBER.4.annotated/7.0.P0_1C14D905.160...  5412386066016



```python
## test data

import os
import pandas as pd

test_annotations_df = pd.DataFrame(columns = ['file_name', 'text'])
test_folders = [folder for folder in os.listdir(test_data_path) if os.path.isdir(os.path.join(test_data_path, folder))]

for folder in test_folders:
    print(folder)
    gt_folder_path = test_data_path + folder + '/' + 'values.csv'
    gt_folder_df   = pd.read_csv(gt_folder_path)
    
    ## remove entries with non-digit labels (illegible images)
    gt_folder_df = gt_folder_df.loc[gt_folder_df['text']!= '_']
  
    
    ## append folder path to image paths
    gt_folder_df['file_name'] = folder + '/' + gt_folder_df['file_name']

    
    # concatenate folder annotatoins to train annotations
    test_annotations_df = pd.concat([test_annotations_df, gt_folder_df])
    

print(test_annotations_df.shape)
print(test_annotations_df.head())

gt_test_path = test_data_path + 'gt-test.txt'
test_annotations_df.to_csv(gt_test_path, header=None, index = None, sep = '\t', mode='a')
```

    PRODUCT_NUMBER.5.annotated
    (199, 2)
                                               file_name           text
    0  PRODUCT_NUMBER.5.annotated/7.0.P2_1B15D905.160...  6414671011144
    1  PRODUCT_NUMBER.5.annotated/8.0.P0_1C14D905.160...  4007841550318
    2  PRODUCT_NUMBER.5.annotated/12.0.P0_1C14D905.16...  5701092112880
    3  PRODUCT_NUMBER.5.annotated/12.0.P2_1B15D905.16...  6438168092263
    4  PRODUCT_NUMBER.5.annotated/15.0.P2_1B15D905.16...  2050004914078



```python

```
