import pandas as pd
import sys
import numpy as np
import os
import pickle

class SVMCFacade(object):
    def __init__(self, svm_model = None):
        self.model = svm_model

    def load_model(self, model_name):
        model_path = os.path.join('.','train', 'output', 'models', 'svm', model_name)

        with open(model_path, 'rb') as input:
            self.model = pickle.load(input)

        print('Model ' + model_name + ' loaded!')
        return self #so we can chain it with get_model
    
    def save_model(self, svm_model, model_name):
        model_path = os.path.join('.','train', 'output', 'models', 'svm', model_name + '.pkl')

        with open(model_path, 'wb') as output:
            pickle.dump(svm_model, output)
        print('Model Pickled at ' + model_path)

    def predict(self, X):
        if self.model is None:
            print('Please load a model!')
        return self.model.predict(X)
    
    def get_model(self):
        return self.model