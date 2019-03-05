'''
    This module contains all necesary functions to perform clustering on an array descriptors, 
    to serialize the resulting clustering model, and to predict bow descriptors based on the model generated
'''
import faulthandler
faulthandler.enable()
import cv2
import numpy as np
import os
from data import preprocessing

class BOWExtractor:
    def __init__(self, vocabulary):
        sift = cv2.xfeatures2d.SIFT_create()
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks=50)   # or pass empty dictionary
        self.extractor = cv2.BOWImgDescriptorExtractor(
            sift, cv2.FlannBasedMatcher(index_params,search_params))
        self.extractor.setVocabulary(vocabulary)
        self.number_of_words = vocabulary.shape[0]

    def compute(self,image, keypoints):
        '''
            using the sift keypoint and descriptor extractor and a bag of features matcher 
            extract the histogram that describes the keypoints for the current image
            we need to use the vocabulary passed as an argument (numpy f32 array)
        '''
        if len(keypoints) != 0:
            bow_descriptor = self.extractor.compute(image, keypoints)
        else:
            bow_descriptor = None
        return self._compute_prior_histogram(self.number_of_words, len(keypoints), bow_descriptor=bow_descriptor)

    def _compute_prior_histogram(self,num_words, keypoint_count, bow_descriptor=None, prior=1):
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


def sift_keypoints(image):
    '''Get list of detected keypoint objects from an image'''
    sift_obj = cv2.xfeatures2d.SIFT_create()
    keypoints = sift_obj.detect(image, None)
    return keypoints

        
def train_knn_model(X, num_clusters=25, trained_model_path=None):

    bow_obj = cv2.BOWKMeansTrainer(num_clusters)

    bow_dict = bow_obj.cluster(X)

    if trained_model_path != None:
        np.save(trained_model_path, bow_dict)
        print('Saved BOWKmeans Dictionary!')
    return bow_dict

def get_image_and_keypoints(image_path):
    image = preprocessing.read_img(image_path)
    return image, sift_keypoints(image)




if __name__ == "__main__":
    extractor = BOWExtractor(vocabulary)

