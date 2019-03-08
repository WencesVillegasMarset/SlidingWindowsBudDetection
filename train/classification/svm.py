import pandas as pd
import sys
import numpy as np
import os
import pickle

class SVMCFacade(object):
    def __init__(self, svm_model = None):
        self.model = svm_model

    def load_model(self, model_name):
        model_path = os.path.join('.', 'output', 'models', 'svm', model_name)

        with open(model_path, 'rb') as input:
            self.model = pickle.load(input)

        print('Model ' + model_name + ' loaded!')
        return self
    
    def save_model(self, svm_model, model_name):
        model_path = os.path.join('.', 'output', 'models', 'svm', model_name + '.pkl')

        with open(model_path, 'wb') as output:
            pickle.dump(svm_model, output)
        print('Model Pickled at ' + model_path)

    def predict(self, X):
        if self.model is None:
            print('Please load a model!')
        return self.model.predict(X)
    
    def get_model(self):
        return self.model

def load_bow_dataset(csv_path, R=1):
    ''' 
        From a csv with columns bow_descriptor (.npy paths), and labels
        get an (X, y) pair containing training or testing data with labels.
        With R=0 we supress sampling functionality
    '''

    csv_file = pd.read_csv(csv_path)
    label_list = []
    descriptor_array_list = []
    if R>=1:
        csv_list = []
        csv_true = csv_file.loc[csv_file['label'] == True, :]
        for i in range(R):
            csv_list.append(csv_true)
        num_samples = csv_true.shape[0] * R
        csv_false = csv_file.loc[csv_file['label'] == False, :]
        csv_false = csv_false.sample(num_samples, random_state = 1)
        csv_list.append(csv_false)
        csv_sampled = pd.concat(csv_list, axis=0)
    else:
        csv_sampled = csv_file


    csv_length = csv_sampled.shape[0]
    i = 0
    for idx, row in csv_sampled.iterrows():
        i = i + 1 
        sys.stdout.write('\rLoading sample ' + str(i) +
                         ' from ' + str(csv_length))
        sys.stdout.flush()
        # read descriptor array from disk
        descriptor_array = np.load(row['bow_descriptor'])
        # append it to the descriptor list
        descriptor_array_list.append(descriptor_array)
        if (row['label'] == True):
            label_list.append(1)
        elif(row['label'] == False):
            label_list.append(0)
        else:
            print('WE ARE IN TROUBLE')
    sys.stdout.write('\nLoading finished!')
    sys.stdout.flush()
    return np.concatenate(descriptor_array_list, axis=0), np.asarray(label_list)

def shuffle_dataset(a, b):
    rng_state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(rng_state)
    np.random.shuffle(b)
    return a, b

def sample_from_dataset(X, y, R=1):
    true_indices = np.where(y == True)
    labels_sampled = []
    descriptors_sampled = []
    for idx in true_indices[0]:
        descriptors_sampled.append(np.reshape(X[idx,:],(1,25)))
        labels_sampled.append(y[idx])
    num_samples = true_indices[0].shape[0]
    false_sampling = np.random.choice(np.where(y == False)[0], size=num_samples)
    print(false_sampling)
    for idx in false_sampling:
        descriptors_sampled.append(np.reshape(X[idx,:],(1,25))) 
        labels_sampled.append(y[idx])

    return np.concatenate(descriptors_sampled, axis=0), np.asarray(labels_sampled)

if __name__ == "__main__":
    os.chdir('..')
    X, y = load_bow_dataset('./output/descriptors/bow/svm_train_set/bow_svm_train_set.csv')
    xs, ys = sample_from_dataset(X=X, y=y)
    