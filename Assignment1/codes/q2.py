#!/usr/bin/env python
import pandas as pd 
import pickle
import numpy as np
import random
import pdb
import os
import sys
def load_data(path):
    df = pd.read_csv(path, header=None)
    df = df.apply(pd.to_numeric, errors = 'coerce')

    columns_list = [i for i in range(1,11)]
    for column in columns_list:
        #avg_val = np.mean(df[column])
        df[column]= df[column].fillna(np.mean(df[column]))
        df[column] = df[column].astype(int)
    X = df.iloc[:,1:-1]
    #y = df.iloc[:,-1]
    y = df.iloc[:, -1].map( {2: 0, 4: 1} ).astype(int)
    return(X, y)

def load_test_data(path):
    df = pd.read_csv(path, header=None)
    df = df.apply(pd.to_numeric, errors = 'coerce')

    columns_list = [i for i in range(1,10)]
    for column in columns_list:
        #avg_val = np.mean(df[column])
        df[column] = df[column].fillna(np.mean(df[column]))
        df[column] = df[column].astype(int)
    X = df.iloc[:,1:] 
    return X


def split_data(X, y, fr= 0.8):
    split_index = int(fr*X.shape[0])
    X_train = np.array([X.iloc[i,:] for i in range(0,split_index)])
    y_train = np.array([y.iloc[i] for i in range(0,split_index)])
    X_val = np.array([X.iloc[i,:] for i in range(split_index, X.shape[0])])
    y_val = np.array([y.iloc[i] for i in range(split_index, X.shape[0])])

    return X_train, y_train, X_val, y_val

def shuffle_data(X,y):
    s = [i for i in range(X.shape[0])]
    X_shuffled = np.zeros((X.shape[0], X.shape[1]), dtype=np.int)
    y_shuffled = np.zeros(len(y), dtype=np.int)
    random.shuffle(s)
    for i in range((X.shape[0])):
        X_shuffled[i,:] = X[s[i],:]
        y_shuffled[i] = y[s[i]]
    return X_shuffled, y_shuffled
def get_weights(input_dimension, output_dimension):
    #print(input_dimension, output_dimension)
    W = np.random.randn(input_dimension, output_dimension)
    #W = np.zeros((input_dimension, output_dimension))
    return W
def sigmoid(X):
    return 1.0 / (1.0 + math.e ** (-1.0 * X)) 
def update(W, X_vec, n):
    W_new = W + n*X_vec
    return W_new
def save(W):
    with open('Weights.pickle', 'wb') as f:
        pickle.dump(W, f)


def accuracy(y_pred, y_val):
    correct= 0
    correct = np.sum(y_pred == y_val)
    accuracy = (correct/float(len(y_val)))*100
    return accuracy

def get_confusion_matrix(y_pred, y_val):
    m = np.zeros((2, 2), dtype = int)
    for pred, exp in zip(y_pred, y_val):
        m[pred][exp] += 1
    return m
def f1_score(y_pred, y_val):
    conf_mat = get_confusion_matrix(y_pred, y_val)
    tp, fp, tn, precison, recall, f_score  = 0, 0, 0, 0.0, 0.0, 0.0
    rows, cols = conf_mat.shape[0], conf_mat.shape[1]
    for i in range(rows):
        fp += sum(conf_mat[i, :])- conf_mat[i, i]
        tp += conf_mat[i, i]
        tn += sum(conf_mat[:, i])- conf_mat[i, i]
        precison = tp/float(fp+tp)
        recall = tp/float(tp+tn)
        f_score = 2*(precison*recall)/(precison+recall)
    return precison, recall, f_score

def augment_data(X_train, y_train):
    f = lambda x,y: x*(-1) if y == 0 else x
    for i in range(X_train.shape[0]):
        
        X_train[i,:] = f(X_train[i,:],y_train[i])
        #pdb.set_trace()
    return X_train, y_train
def validation(X_val, y_val, W, margin):
    threshold = 0
    y_pred =[]
    for i in range(X_val.shape[0]):    
        if np.dot(X_val[i], np.transpose(W)) >= margin:
            y_pred.append(1)
        else:
            y_pred.append(0)
    acc = accuracy(y_pred, y_val)
    precison, recall, f_score = f1_score(y_pred, y_val)
    return acc, precison, recall, f_score


def train(X_train, y_train, X_val, y_val, margin, eta):
    flag, su, epoch,theta,count  = 1, 5.001, 0, 0.0001,1
    errors=1
    W  = get_weights(1, X_train.shape[1])
    W_prev = np.zeros((1, X_train.shape[0]), dtype= np.float)
    row = []
    while(epoch<=30):
        flag =0
        errors =0
        su =0
        c=0
        diff= 0
        for i in range(X_train.shape[0]):
            if np.dot(X_train[i,:], np.transpose(W)) <= margin :
                c+=1
                norm_X = np.linalg.norm(X_train[i, :])
                factor = (margin - np.dot(X_train[i,:], np.transpose(W)))/(norm_X**2)
                X_vec = (X_train[i].astype(np.float64))*factor   
                W_prev = W
                W = update(W, X_vec, eta)
                su += 0.5*((margin - np.dot(X_train[i,:], np.transpose(W)))**2/(norm_X**2))
        diff = [each_element for each_element in np.dot(X_train, np.transpose(W))]
        errors = sum(list(map(lambda x: 1 if x <= margin else 0, diff)))

        acc, precison, recall, f_score = validation(X_val, y_val, W, margin)
        row.append([epoch, acc, precison, recall, f_score])
        epoch+=1
        eta += 0.09

    #print('Saving Weights..')
    
    save(W)
    return W, row

def test(X_test, W, margin):
    threshold = 0
    y_pred =[]
    for i in range(X_val.shape[0]):    
        if np.dot(X_val[i], np.transpose(W)) >= margin:
            sys.stdout.write(str(4) + '\n')
        else:
            sys.stdout.write(str(2) + '\n')


if __name__ == '__main__':
    train_path = sys.argv[1]
    test_path = sys.argv[2]
    X, y = load_data(train_path)
    X_test = load_test_data(test_path)
    X_train, y_train, X_val, y_val = split_data(X, y)
    X_train, y_train = shuffle_data(X_train, y_train)
    X_train, y_train = augment_data(X_train, y_train)
    try:

        Weights, row = train(X_train, y_train,X_val, y_val, margin=44, eta=0.5)
        df = pd.DataFrame(row, columns=['Epoch', 'Accuracy', 'Precision', 'Recall', 'f1_score'])
        df.to_csv('single_perceptron_relaxation.csv')
        test(X_test, Weights, margin=44)
    except KeyboardInterrupt:
        print("Saving before quit...")
        save(Weights)
    #X_val, y_val = augment_data(X_val, y_val)
    #validation(X_val, y_val, Weights, margin)