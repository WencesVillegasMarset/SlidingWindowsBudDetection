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
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)   # or pass empty dictionary
    self.extractor = cv2.BOWImgDescriptorExtractor(
        sift, cv2.FlannBasedMatcher(index_params,search_params))
    bow_ext.setVocabulary(vocabulary)
    if len(keypoints) != 0:
        bow_descriptor = bow_ext.compute(image, keypoints)
    else:
        bow_descriptor = None
  
    return _compute_prior_histogram(25, len(keypoints), bow_descriptor)

def _compute_prior_histogram(num_words, keypoint_count, bow_descriptor=None, prior=1):
        '''
            Once a bow descriptor for an images sift descriptor has been calculated we normalize its values 
            given a prior value (1)
            If the bow_descriptor doesnt exist because there we no keypoints detected we generate a default normalized 
            descriptor.
        '''
        base_prior = prior/num_words
        base_descriptor = np.ones((1, num_words), dtype=np.float32) * base_prior
        if bow_descriptor is None:
            return base_descriptor
        else:
            # scale the histogram values by the number of keypoints
            bow_descriptor = bow_descriptor * keypoint_count
            kp_scale = keypoint_count + prior
            return (bow_descriptor + base_descriptor) / np.float32(kp_scale)

def python_svm_predict():
    pass

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