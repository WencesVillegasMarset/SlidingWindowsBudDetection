import pandas as pd
import matplotlib.pyplot as plt
import os
import cv2
import numpy as np


csv = pd.read_csv('../output/result_300_150step_04ovlp.csv')
IMAGES_PATH = '/home/wences/Documents/src/images/images/'
PROJECTIONS_PATH = '../output/300_150step_04ovlp/projections/'
MASKS_PATH = '../output/300_150step_04ovlp/masks/'
ARRAYS_PATH = '../output/300_150step_04ovlp/vote_arrays/'
BASE_IMAGES_PATH = '/home/wences/Documents/corpus-26000/images/'
BINARY_MASK_PATH = '../output/300_150step_04ovlp/binary_masks/'

positive_patches = csv.loc[csv['svm_result'] > 0.5,:]

list_positive_images = positive_patches['image_name'].unique()

for image_name in list_positive_images:
    patches_from_image = positive_patches.loc[positive_patches['image_name'] == image_name,:]
    img = cv2.cvtColor(cv2.imread(IMAGES_PATH + image_name), cv2.COLOR_BGR2RGB)
    for idx, row in patches_from_image.iterrows():
        cv2.rectangle(img,(row['top_left_corner_y'],row['top_left_corner_x']),(row['top_left_corner_y']+300,row['top_left_corner_x']+300),(0,255,0),int(row['svm_result']*35))
    cv2.imwrite(PROJECTIONS_PATH + 'sw_'+ image_name, img)

for image_name in list_positive_images:
    patches_from_image = positive_patches.loc[positive_patches['image_name'] == image_name,:]
    img = cv2.cvtColor(cv2.imread(IMAGES_PATH + image_name), cv2.COLOR_BGR2RGB)
    mask = np.zeros_like(img[:,:,0])
    for idx, row in patches_from_image.iterrows():
        x = row['top_left_corner_x']
        y = row['top_left_corner_y']
        mask[x:x+300,y:y+300] += 1
        #cv2.rectangle(img,(row['top_left_corner_x'],row['top_left_corner_y']),(row['top_left_corner_x']+300,row['top_left_corner_y']+300),(0,255,0),20)
    cv2.imwrite(MASKS_PATH + 'mask_sw_'+ image_name, cv2.normalize(mask,None,0,255,cv2.NORM_MINMAX))

for image_name in list_positive_images:
    patches_from_image = positive_patches.loc[positive_patches['image_name'] == image_name,:]
    img = cv2.cvtColor(cv2.imread(IMAGES_PATH + image_name), cv2.COLOR_BGR2RGB)
    mask = np.zeros_like(img[:,:,0])
    for idx, row in patches_from_image.iterrows():
        x = row['top_left_corner_x']
        y = row['top_left_corner_y']
        mask[x:x+300,y:y+300] += 1
    np.save(ARRAYS_PATH + 'mask_sw_'+ image_name[0:4]+'.npy', mask)

npy_mask = os.listdir(ARRAYS_PATH)
npy_mask = sorted(npy_mask)

for npy_array in npy_mask:
    mask = np.load(ARRAYS_PATH + npy_array)
    mask[mask<3] = 0
    mask = np.uint8(mask.astype(bool))
    cv2.imwrite(BINARY_MASK_PATH + 'bin_' + npy_array[0:-4] + '.jpg', cv2.normalize(mask,None,0,255,cv2.NORM_MINMAX))

