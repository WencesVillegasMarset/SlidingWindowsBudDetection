from clustering import cluster_bow
from data import preprocessing
import pandas as pd
import os
import numpy as np
from classification import svm
from sklearn.svm import SVC

def fit_knn(train_csv_path, vocabulary_output_path):
    train_csv = pd.read_csv(train_csv_path)
    image_list = train_csv['imageName'].values
    label_list = train_csv['class'].values
    tuple_list = zip(image_list, label_list)
    out_csv = os.path.join('.', 'output', 'descriptors', 'sift',
                           os.path.split(train_csv_path)[1])
    out_descriptors = os.path.join('.', 'output', 'descriptors', 'sift', preprocessing.remove_extension(
        os.path.split(train_csv_path)[1]))

    if not os.path.exists(out_csv):
        preprocessing.generate_descriptor_file(
            tuple_list, out_csv, out_descriptors)

    X, y = preprocessing.load_serialized_dataset(out_csv)
    print('\nClustering dataset!')

    vocabulary = cluster_bow.train_knn_model(
        X, trained_model_path=vocabulary_output_path)
    print('\nClustering Finished!')
    return vocabulary


def generate_bow_descriptors(vocabulary, csv_path):

    dataset = pd.read_csv(csv_path)
    bow_obj = cluster_bow.BOWExtractor(vocabulary)
    bow_descriptor_list = []
    label_list = []
    output_csv_path = os.path.join('.', 'output', 'descriptors', 'bow', preprocessing.remove_extension(
        os.path.split(csv_path)[1]), 'bow_'+os.path.split(csv_path)[1])
    if not os.path.exists(os.path.join('.', 'output', 'descriptors', 'bow', preprocessing.remove_extension(
            os.path.split(csv_path)[1]))):
        os.makedirs(os.path.join('.', 'output', 'descriptors', 'bow', preprocessing.remove_extension(
            os.path.split(csv_path)[1])))

    if os.path.exists(output_csv_path):
        return output_csv_path

    for idx, row in dataset.iterrows():
        img = preprocessing.read_img(row['imageName'])
        keypoints = cluster_bow.sift_keypoints(img)
        bow_desc = bow_obj.compute(img, keypoints)
        bow_desc_path = os.path.join('.', 'output', 'descriptors', 'bow', preprocessing.remove_extension(
            os.path.split(csv_path)[1]), preprocessing.remove_extension(os.path.split(row['imageName'])[1]))
        print(bow_desc_path)
        np.save(bow_desc_path, bow_desc)
        bow_descriptor_list.append(bow_desc_path + '.npy')
        label_list.append(row['class'])

    dataframe = {
        'bow_descriptor': bow_descriptor_list,
        'label': label_list
    }
    dataframe = pd.DataFrame(dataframe)
    dataframe.to_csv(output_csv_path)
    return output_csv_path


if __name__ == "__main__":
    from sklearn.metrics import classification_report
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import f1_score

    vocabulary = np.load(
        '/home/wences/Documents/GitRepos/SlidingWindowsBudDetection/train/output/models/bow/svm_train_set.npy')
    #generate_bow_descriptors(vocabulary, './resources/svm_test_set.csv')
    #generate_bow_descriptors(vocabulary, './resources/svm_train_set.csv')
    X_train, y_train = svm.load_bow_dataset('/home/wences/Documents/GitRepos/SlidingWindowsBudDetection/train/output/descriptors/bow/svm_train_set/bow_svm_train_set.csv')
    X_train, y_train = svm.shuffle_dataset(X_train, y_train)
    X_test, y_test = svm.load_bow_dataset('/home/wences/Documents/GitRepos/SlidingWindowsBudDetection/train/output/descriptors/bow/svm_test_set/bow_svm_test_set.csv', R=1)
    
    exp_c = np.linspace(7,12,10)
    c_values = [2**k for k in exp_c]
    exp_g = np.linspace(4,9,10)
    gammma_values = [2**(-k) for k in exp_g]
    kfold_scores = []
    for c in c_values:
        for gamma in gammma_values:
            svm = SVC(C=c, kernel='rbf', gamma=gamma, random_state=1)
            scores = cross_val_score(svm, X_train, y_train, cv=5, scoring='f1')
            kfold_scores.append(scores.mean())
    print(kfold_scores)
    print(max(kfold_scores))



