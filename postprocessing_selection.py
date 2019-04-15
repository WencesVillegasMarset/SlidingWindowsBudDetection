import pandas as pd
import matplotlib.pyplot as plt
import os
import cv2
import numpy as np
import argparse
def read_image_grayscale(path):
    return cv2.imread(path, 0)
def get_sample_ground_truth(sample_name, csv):
    name = (sample_name.split('.')[0]).split('_')[-1] + '.jpg'
    return (csv.loc[csv['imageOrigin'] == name, :])
def connected_components(img):
    num_components, labeled_img =  cv2.connectedComponents(img, connectivity=8)
    label_array = np.arange(0,num_components)[1::]
    return label_array, labeled_img

def cluster_mass_center(mask, labels):
    if labels.shape[0] == 0:
        return np.nan
    mass_center_array = []
    for label in labels:
        cluster_array = (mask == label)
        mass_center_array.append(utils_cluster.mass_center(cluster_array.astype(int)))
    return (np.asarray(mass_center_array))

def run(args):

    masks_folder = args.path

    mask_list = os.listdir(masks_folder)

    ground_truth_csv = pd.read_csv('./corpus-26000-positivo.csv')

    metrics = {
        'model_name':[],
        'mask_name':[],
        'eps':[],
        'min_samples':[],
        'buds_predicted':[],
        'true_positive_x':[],
        'true_positive_y':[],
        'true_positive_distance':[]
    }
    for mask in mask_list:

    

def main():
    parser = argparse.ArgumentParser(
        description="Run postprocessing parameter selection for SW")

    parser.add_argument("-path", help="path for binary masks",
                        dest="csv", type=str, required=True)

    parser.set_defaults(func=run)

    args = parser.parse_args()

    if (not os.path.exists(args.csv)):
        parser.error('Invalid path to csv')
    
    args.func(args)


if __name__ == "__main__":
    main()