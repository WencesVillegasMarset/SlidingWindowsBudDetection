import yaml
import numpy as np
import cv2
import subprocess
import os
from subprocess import PIPE

# we need a custom constructor to interpret the opencv.matrix format specified in the yml file


def opencv_matrix(loader, node):
    '''
        function used as a callback to read data from yml file and format it to a matrix 
        that opencv can use to compute bow descriptors
    '''
    mapping = loader.construct_mapping(node, deep=True)
    mat = np.array(mapping["data"])
    mat.resize(mapping["rows"], mapping["cols"])
    return mat.astype(np.float32)


def read_bof(path):
    '''
        read bag_of_features parameters from yml file and return a dictionary with the same structure as the yml file
    '''
    yaml.add_constructor(u"tag:yaml.org,2002:opencv-matrix", opencv_matrix)
    with open(path, 'r') as stream:
        try:
            return (yaml.load(stream))
        except yaml.YAMLError as exc:
            print(exc)


def compute_bow_histogram(image, keypoints, vocabulary=None):
    '''
        using the sift keypoint and descriptor extractor and a bag of features matcher 
        extract the histogram that describes the keypoints for the current image
        we need to use the vocabulary loaded previously from an external file
        If vocabulary isnt passed we load the deafault yml file hardcoded in the if statement
    '''
    if vocabulary == None:
        bof_yaml_dict = read_bof('dict-s25.yml')
        vocabulary = bof_yaml_dict['vocabulary']
    sift = cv2.xfeatures2d.SIFT_create()
    bow_ext = cv2.BOWImgDescriptorExtractor(sift, cv2.BFMatcher(cv2.NORM_L2))
    bow_ext.setVocabulary(vocabulary)
    return (bow_ext.compute(image, keypoints))



def run_svm_script(rdata_file_path, descriptor):
    '''
        get the probability that a patch has a bud in it
        run an R function in the command line and save the output, then process it to return a probability value
    '''
    command = "R -q -e \"source(\'"+rdata_file_path+"\');patchClassifier(c("+ ','.join([str(e) for e in descriptor]) +"))\""
    output = subprocess.check_output(command, shell=True)
    return float((output.split()[3]).decode("utf-8"))

if __name__ == "__main__":
    from helpers import image
    import sift
    bof_dict = read_bof('dict-s25.yml')
    img = image.read_img('0001.jpg')
    descriptor = (compute_bow_histogram(img, sift.sift_keypoints(img)))
    probability = run_svm_script("/home/wences/Documents/GitRepos/SlidingWindowsBudDetection/svm.R",descriptor[0])
    print(probability)