from clustering import cluster_bow
from data import preprocessing
import pandas as pd
import os
import numpy as np
from classification import svm
from sklearn.svm import SVC
import matplotlib.pyplot as plt


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
        image_name = os.path.split(row['imageName'])[1]
        img = preprocessing.read_img(row['imageName'])
        keypoints = cluster_bow.sift_keypoints(img)
        bow_desc = bow_obj.compute(img, keypoints)
        bow_desc_path = os.path.join('.', 'output', 'descriptors', 'bow', preprocessing.remove_extension(
            os.path.split(csv_path)[1]), preprocessing.remove_extension(image_name))
        print(image_name)
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


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import f1_score
    import itertools

    vocabulary = np.load(
        '/home/wences/Documents/GitRepos/SlidingWindowsBudDetection/train/output/models/bow/svm_train_set.npy')
    #generate_bow_descriptors(vocabulary, './resources/svm_test_set.csv')
    #generate_bow_descriptors(vocabulary, './resources/svm_train_set.csv')
    X_train, y_train = svm.load_bow_dataset(
        '/home/wences/Documents/GitRepos/SlidingWindowsBudDetection/train/output/descriptors/bow/svm_train_set/bow_svm_train_set.csv')
    X_train, y_train = svm.shuffle_dataset(X_train, y_train)
    X_test, y_test = svm.load_bow_dataset(
        '/home/wences/Documents/GitRepos/SlidingWindowsBudDetection/train/output/descriptors/bow/svm_test_set/bow_svm_test_set.csv', R=0)

    '''exp_c = np.linspace(7,12,10)
    c_values = [2**k for k in exp_c]
    exp_g = np.linspace(4,9,10)
    gammma_values = [2**(-k) for k in exp_g]
    kfold_scores = []
    best_score = None
    for c in c_values:
        for gamma in gammma_values:
            svm_cv = SVC(C=c, kernel='rbf', gamma=gamma, random_state=1)
            scores = cross_val_score(svm_cv, X_train, y_train, cv=5, scoring='f1')
            kfold_scores.append(scores.mean())
            if scores.mean() == max(kfold_scores):
                best_score = svm_cv
    print(kfold_scores)
    print(max(kfold_scores))
    svm_services = svm.SVMCFacade()
    best_score.fit(X_train, y_train)
    svm_services.save_model(best_score, 'best_svm_cv_1')'''

    svm_services = svm.SVMCFacade()
    best_score = svm_services.load_model('best_svm_cv_1.pkl').get_model()
    y_pred = best_score.predict(X_test)
    print(classification_report(y_test, y_pred,
                                target_names=['no-yema', 'yema']))
    plot_confusion_matrix(confusion_matrix(y_test, y_pred), classes=['noyema','yema'])    