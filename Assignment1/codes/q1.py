import pandas as pd 
import pickle
import numpy as np
import random
import pdb
import os
import sys
def load_data(path):
    df = pd.read_csv(path)
    X = df.iloc[:,1:]
    y = df.iloc[:,0]
    return(X, y)
def load_test_data(path):
    df = pd.read_csv(path)
    X = df.iloc[:,:]
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



def validation(X_val, y_val, W):
    threshold = 0
    y_pred =[]
    for i in range(X_val.shape[0]):    
        if np.dot(X_val[i], np.transpose(W)) >= threshold:
            y_pred.append(1)
        else:
            y_pred.append(0)
    acc = accuracy(y_pred, y_val)
    precison, recall, f_score = f1_score(y_pred, y_val)
    return acc, precison, recall, f_score

def train(X_train, y_train, margin, eta):
    flag, su, errors, epoch, theta = 1, 1, 1, 0, 0.0001
    row = []
    W  = get_weights(1, X_train.shape[1])

    while(errors > theta):
        flag,su, errors,val_accuracy = 0, 0, 0, 0.0 
        for i in range(X_train.shape[0]):
            if np.dot(X_train[i,:], np.transpose(W)) <= margin :
                flag = 1
                W = update(W, X_train[i], eta)
                
                    
        
        diff = [each_element for each_element in np.dot(X_train, np.transpose(W))]
        errors = sum(list(map(lambda x: 1 if x <= margin else 0, diff)))

        epoch+=1

        acc, precison, recall, f_score = validation(X_val, y_val, W)
        row.append([epoch, acc, precison, recall, f_score])
        #print('Epoch %d, Loss: %.4f'%(epoch, errors))
    #print('Saving Weights..')
    
    save(W)
    return W, row
def train_batch(X_train, y_train, margin, eta):
    flag, su, errors, theta, epoch  = 1, 1, 1, 0.0001, 0
    row = []
    W = get_weights(1, X_train.shape[1])
    while(su > theta or epoch < 500):
        flag, su, errors, val_accuracy = 0, 0, 0, 0.0
        Xsum =[]
        Xk  = np.dot(X_train, np.transpose(W))
        indices = [i for i, e in enumerate(list(map(lambda t: t if t <= margin else 0, Xk))) if e != 0]
        Xsubset = [X_train[indices[i],:] for i in range(len(indices))]
        Xsubset = np.array(Xsubset)
        #pdb.set_trace()
        Xsum = np.sum(Xsubset, axis=0)
        #pdb.set_trace()
        W = update(W, Xsum, eta)
        flag =1
        
        for i in range(Xsubset.shape[0]):
            
            su+= abs(np.dot(Xsubset[i], np.transpose(W)))
           
        
        diff = [each_element for each_element in np.dot(X_train, np.transpose(W))]
        errors = sum(list(map(lambda x: 1 if x <= margin else 0, diff)))
        epoch+=1
        acc, precison, recall, f_score = validation(X_val, y_val, W)
        row.append([epoch, acc, precison, recall, f_score])
        #print('Epoch %d, Loss: %.4f'%(epoch, su))
    save(W)
    return W, row


   
def test(X_test, W):
    threshold = 0
    y_pred =[]
    for i in range(X_val.shape[0]):    
        if np.dot(X_val[i], np.transpose(W)) >= threshold:
            sys.stdout.write(str(1) + '\n')
        else:
            sys.stdout.write(str(0) + '\n')

    
if __name__ == '__main__':
    train_path = sys.argv[1]
    test_path = sys.argv[2]

    #margin = 0
    #eta = 1
    X, y = load_data(train_path)
    X_test = load_test_data(test_path)
    X_train, y_train, X_val, y_val = split_data(X, y)
    X_train, y_train = shuffle_data(X_train, y_train)
    X_train, y_train = augment_data(X_train, y_train)
    try:
        Weights, row = train(X_train, y_train, margin= 0, eta=1)
        df = pd.DataFrame(row, columns=['Epoch', 'Accuracy', 'Precision', 'Recall', 'f1_score'])
        df.to_csv('single_perceptron.csv')
        test(X_test, Weights)
        Weights, row = train(X_train, y_train, margin =15, eta=1)
        df = pd.DataFrame(row, columns=['Epoch', 'Accuracy', 'Precision', 'Recall', 'f1_score'])
        df.to_csv('single_perceptron_margin.csv')
        test(X_test, Weights)
        Weights, row = train_batch(X_train, y_train, margin=0, eta=1)
        df = pd.DataFrame(row, columns=['Epoch', 'Accuracy', 'Precision', 'Recall', 'f1_score'])
        df.to_csv('batch_perceptron.csv')
        test(X_test, Weights)
        Weights, row = train_batch(X_train, y_train, margin=5, eta=1)
        df = pd.DataFrame(row, columns=['Epoch', 'Accuracy', 'Precision', 'Recall', 'f1_score'])
        df.to_csv('batch_perceptron_margin.csv')
        test(X_test, Weights)
    except KeyboardInterrupt:
        print("Saving before quit...")
        save(Weights)
    
    