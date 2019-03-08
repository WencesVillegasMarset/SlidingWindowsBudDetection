import argparse
import os
import numpy as np
import helpers
import helpers.image as im
import helpers.patch as patch
import sift
import classifier
import pandas as pd
import time
from svm import SVMCFacade
def run(args):
    '''
        Main 
    '''

    img_csv_path = args.img
    patch_size = args.size
    step = args.step
    output_path = args.out
    bag_of_features_path = args.bof
    svm_param_path = args.svm
    ground_truth_csv_path = args.gt
    min_overlap = args.ovlp

    csv_data_path = os.path.join(output_path, 'image_csv_data')
    keypoints_data = os.path.join(output_path, 'keypoints_data')

    image_list = pd.read_csv(img_csv_path, header=None)
    image_list = image_list.iloc[:, 0].values

    ground_truth_csv = pd.read_csv(ground_truth_csv_path)
    image_csv_data = {
                'image_name': [],
                'top_left_corner_x': [],
                'top_left_corner_y': [],
                'ground_truth': [],
                'svm_result': [],
                'keypoint_info': []
            }

    #prepare SVMFacade to perform prediction
    svm_facade = SVMCFacade()
    svm_obj = svm_facade.load_model('best_svm_cv_1.pkl').get_model()



    for img in image_list:
        img_name = os.path.split(img)[1]
        print('Processing: ' + img_name)
        
        # read image
        start_time = time.clock()

        bud_img = im.read_img(img)
        # get patch top-left coordinates for this image
        patch_coordinates = patch.sliding_window(
            bud_img, patch_size, step)
        for p in range(patch_coordinates.shape[0]):
            image_csv_data['image_name'].append(img_name)
            image_csv_data['top_left_corner_x'].append(patch_coordinates[p, 0])
            image_csv_data['top_left_corner_y'].append(patch_coordinates[p, 1])
            #get slice of the image array for this current patch
            current_patch = patch.get_patch(
                bud_img, patch_coordinates[p, :], patch_size)
            #add the ground truth for this patch

            ################### NOT USEFUL CODE
            #image_csv_data['ground_truth'].append(patch.search_for_ground_truth(
                #ground_truth_csv, img_name, patch_coordinates[p, 0], patch_coordinates[p, 1], patch_size, min_overlap))


            #get descriptor of current patch passing the current patch and its keypoints
            kp = sift.sift_keypoints(current_patch)
            descriptor = (classifier.compute_bow_histogram(current_patch, kp))
                #get the probability that a bud is present on that patch running the R script
            bud_prescence_probability = svm_obj.predict_proba(descriptor)
            #bud_prescence_probability = classifier.run_svm_script("/home/wences/Documents/GitRepos/SlidingWindowsBudDetection/svm.R",descriptor[0])         
            image_csv_data['svm_result'].append(bud_prescence_probability)
            else:
                image_csv_data['svm_result'].append(0)
            #serialize info for the keypoints of the patch or append none if that isnt necessary
            image_csv_data['keypoint_info'].append('none')
        print("--- %s seconds ---" % (time.clock() - start_time))

    
    dataframe = pd.DataFrame(image_csv_data)
    dataframe.to_csv(os.path.join(output_path, 'result' + str(patch_size) + '_' +str(step) + 'step'+ '_'+str(min_overlap) + 'ovlp' +'.csv'))
    print('Done!')


def main():
    parser = argparse.ArgumentParser(
        description="Run Sliding Windows Detection on bud images")
    parser.add_argument("-img", help="csv containing absolute paths for test set images  (1 Column)",
                        dest="img", type=str, required=True)
    parser.add_argument("-size", help="size of square patch",
                        dest="size", type=int, required=True)
    parser.add_argument('-step', help="step size in pixels",
                        dest="step", type=int, required=True)
    parser.add_argument("-out", help="path to folder containing the output to be generated",
                        dest="out", type=str, required=True)
    parser.add_argument("-bof", help="bof .yml file path",
                        dest="bof", type=str, required=True)
    parser.add_argument("-svm", help="svm RData file path",
                        dest="svm", type=str, required=True)
    parser.add_argument("-gt", help="ground truth csv path",
                        dest="gt", type=str, required=True)
    parser.add_argument(
        '-ovlp', help="overlap percentage between a gt patch and a generated patch to be considered a positive one [0-1]", dest="ovlp", type=float, required=True)

    parser.set_defaults(func=run)
    args = parser.parse_args()
    if (not os.path.exists(args.img)):
        parser.error('Invalid path to images csv')
    if (not os.path.exists(args.out)):
        parser.error('Invalid path to output folder')
    if (not os.path.exists(args.bof)):
        parser.error('Invalid path to bof .yml file')
    if (not os.path.exists(args.svm)):
        parser.error('Invalid path to svm RData file')
    if (not os.path.exists(args.gt)):
        parser.error('Invalid path to ground truth csv')
    if not (0 < args.ovlp <= 1):
        parser.error('Invalid minimum overlap value')

    args.func(args)


if __name__ == "__main__":
    main()
