import cv2
import numpy as np
import os
import pandas as pd
import sys


def read_img(path):
    '''
        Read an image in RGB mode, full path required
    '''
    return cv2.cvtColor(cv2.imread(path, 1), cv2.COLOR_BGR2RGB)


def remove_extension(filename):
    return filename.split('.')[0]


def save_img(path, image):
    '''
        Save an image in the specified path with the specified imagename e.g (/path/to/, 'mypic.jpg')
    '''
    cv2.imwrite(path, image)


def sift_detect_and_compute(image):
    '''
            Get sift descriptors from image, we not only return the keypoint
            objects but also we compute its corresponding descriptors
    '''
    sift_obj = cv2.xfeatures2d.SIFT_create()
    keypoints, descriptors = sift_obj.detectAndCompute(image, None)
    return (keypoints, descriptors)


def draw_keypoints(image, keypoints):
    '''
            draw keypoints on image
    '''
    return (cv2.drawKeypoints(image, keypoints, cv2.DRAW_MATCHES_FLAGS_DRAW_OVER_OUTIMG))


def generate_descriptor_file(image_list, out_csv, descriptor_path, rgb=True):
    '''
        input a list of tuples that contain image paths and correponding label,
        save descriptor arrays to disk and generate a csv
    '''
    if not os.path.exists(descriptor_path):
        os.makedirs(descriptor_path)
    image_name_list = []
    descriptor_list = []
    label_list = []
    num_keypoint_list = []

    for image_tuple in image_list:
        sys.stdout.write('\rSerializing sample ' + image_tuple[0])
        sys.stdout.flush()
        image_name = os.path.split(image_tuple[0])[1]
        if rgb:
            image_file = read_img(image_tuple[0])
        else:
            pass  # not implemented yet
        keypoints, descriptors = sift_detect_and_compute(image_file)
        if len(keypoints) == 0:
            descriptors = np.ones((1, 1))

        num_keypoint_list.append(len(keypoints))
        np.save(os.path.join(descriptor_path,
                             remove_extension(image_name)), descriptors)
        image_name_list.append(image_tuple[0])
        descriptor_list.append(os.path.join(
            descriptor_path, remove_extension(image_name) + '.npy'))
        label_list.append(image_tuple[1])

    print('\nSerialization Complete!')
    dict_csv = {
        'imageName': image_name_list,
        'descriptors': descriptor_list,
        'label': label_list,
        'num_keypoints': num_keypoint_list

    }
    dataframe = pd.DataFrame(dict_csv)
    dataframe.to_csv(out_csv)


def load_serialized_dataset(csv_path):
    ''' 
        From a csv with columns image_name, descriptors (.npy paths), and labels
        get an (X, y) pair containing training or testing data with labels.
    '''
    csv_file = pd.read_csv(csv_path)
    label_list = []
    descriptor_array_list = []

    csv_length = csv_file.shape[0]

    for idx, row in csv_file.iterrows():
        sys.stdout.write('\rLoading sample ' + str(idx) +
                         ' from ' + str(csv_length))
        sys.stdout.flush()
        # read descriptor array from disk
        descriptor_array = np.load(row['descriptors'])
        # append it to the descriptor list
        if descriptor_array.shape[1] == 128:
            descriptor_array_list.append(descriptor_array)
        for desc_row in range(descriptor_array.shape[0]):
            # we need to have a label for each training sample (descriptor row)
            # so we append that same amount of labels to the label list
            label_list.append(row['label'])
    sys.stdout.write('\nLoading finished!')
    sys.stdout.flush()
    return np.concatenate(descriptor_array_list, axis=0), np.asarray(label_list)




if __name__ == "__main__":

    generate_descriptor_file(
        [('/home/wences/Documents/corpus-26000/corpus-26000-bc/0001.jpg', True)], './csv.csv', '.')
    print(get_serialized_dataset('./csv.csv'))
