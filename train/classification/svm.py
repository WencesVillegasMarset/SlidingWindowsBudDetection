import pandas as pd
import sys
import numpy as np

class SVMClassifier(object):
    def __init__(self, parameter_list):
        self.model = None

    def load_model():
        pass

    def predict():
        pass

def load_bow_dataset(csv_path):
    ''' 
        From a csv with columns bow_descriptor (.npy paths), and labels
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
        descriptor_array = np.load(row['bow_descriptor'])
        # append it to the descriptor list
        descriptor_array_list.append(descriptor_array)
        if (row['label'] == True):
            label_list.append(1)
        else:
            label_list.append(0)
    sys.stdout.write('\nLoading finished!')
    sys.stdout.flush()
    return np.concatenate(descriptor_array_list, axis=0), np.asarray(label_list)