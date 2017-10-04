"""
Course: Statistical Methods in Artificial Intelligence (CSE471)
Semester: Fall '17
Professor: Gandhi, Vineet

Assignment 2: SVM using scikit-learn.
Skeleton code for implementing SVM classifier for a
character recognition dataset having precomputed features for
each character. This is your submission file.

Dataset is taken from: https://archive.ics.uci.edu/ml/datasets/letter+recognition

Remember
--------
1) SVM algorithms are not scale invariant.
"""

from sklearn import svm
from sklearn import preprocessing
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold.t_sne import TSNE
import pandas as pd
import argparse, os, sys
import pdb
def get_input_data(filename):
    

    X = []; Y = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip().split(',')
            Y.append(line[0])
            X.append([float(x) for x in line[1:]])
    X = np.asarray(X); Y = np.asarray(Y)

    X = preprocessing.scale(X)

    return X, Y

def calculate_metrics(predictions, labels):
    labels = [ord(l) - 65 for l in labels]
    predictions = [ord(p) - 65 for p in predictions]
    precision, recall, f1 = 0.0, 0.0, 0.0
    num_classes = len(set(labels))
    conf_mat = np.zeros((num_classes, num_classes))
    #pdb.set_trace() 
    
    for pred, exp in zip(predictions, labels):
            conf_mat[pred][exp] += 1
    df  = pd.DataFrame(conf_mat)
    df.to_csv('confusion_mat.csv')
    tp, fp, fn = 0.0, 0.0, 0.0 
    
    for i in range(num_classes):
            
            fp += sum(conf_mat[i, :])- conf_mat[i, i]
            tp += conf_mat[i, i]
            fn += sum(conf_mat[:, i])- conf_mat[i, i]

    precision = tp/float(tp+fp)
    recall = tp/float(tp+fn)
    f1 = 2*(precision*recall)/(precision+recall)
    return precision, recall, f1

    return precision, recall, f1

def calculate_accuracy(predictions, labels):
    return accuracy_score(labels, predictions)

def SVM(train_data,
        train_labels,
        test_data,
        test_labels,
        kernel, alpha):

    clf = svm.SVC(decision_function_shape='ovo', kernel=kernel, C=alpha)
    clf.fit(train_data, train_labels) 
    
    predictions = clf.predict(test_data)
    accuracy = calculate_accuracy(predictions, test_labels)
    precision, recall, f1 = calculate_metrics(predictions, test_labels)
    return accuracy, precision, recall, f1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default=None,
            help='path to the directory containing the dataset file')

    args = parser.parse_args()
    if args.data_dir is None:
        print ("Usage: python letter_classification_svm.py --data_dir='<dataset dir path>'")
        sys.exit()
    else:
        filename = os.path.join(args.data_dir, 'letter_classification_train.data')
        try:
            if os.path.exists(filename):
                print ("Using %s as the dataset file" % filename)
        except:
            print ("%s not present in %s. Please enter the correct dataset directory" % (filename, args.data_dir))
            sys.exit()

    # Set the value for svm_kernel as required.
    svm_kernel = 'rbf'

    X_data, Y_data = get_input_data(filename)
    sss = StratifiedShuffleSplit(n_splits=5, test_size=0.20)   # Do not change this split size
    accumulated_metrics = []
    fold = 1
    for train_indices, test_indices in sss.split(X_data, Y_data):
        #print ("Fold%d -> Number of training samples: %d | Number of testing "\
            #"samples: %d" % (fold, len(train_indices), len(test_indices))
        train_data, test_data = X_data[train_indices], X_data[test_indices]
        train_labels, test_labels = Y_data[train_indices], Y_data[test_indices]
        accumulated_metrics.append(
            SVM(train_data, train_labels, test_data, test_labels,
                svm_kernel, 0.6))
        fold += 1

    accumulated_metrics_mean = (np.array(accumulated_metrics).mean(axis=0))
    print('%.2f, %.2f, %.2f, %.2f'%(accumulated_metrics_mean[0], accumulated_metrics_mean[1],
                                                accumulated_metrics_mean[2], accumulated_metrics_mean[3]))