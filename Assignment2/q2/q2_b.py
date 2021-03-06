import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD, adam
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense, BatchNormalization
from keras.utils import np_utils
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras import initializers
import os
import argparse
import pdb
import pandas as pd
import numpy as np
import theano
from sklearn import svm
import sys
import random
from sklearn.metrics import accuracy_score
os.environ['CUDA_VISIBLE_DEVICE']='0'
os.environ['THEANO_FLAGS']='device=gpu'
seed = 7
numpy.random.seed(seed)

def unpickle(file):
    import pickle

    with open(file, 'rb') as fo:
        mydict = pickle.load(fo)
    return mydict

def load_data(path):
	n = len(os.listdir(path))
	data = np.zeros((n*10000, 3072))
	labels = np.zeros((n*10000), dtype=int)
	for i in range(1,n+1):
		data_dict = unpickle(os.path.join(path,'data_batch_%d'%i))
		row = (i-1)*10000
		data[row:row+10000, :] = data_dict['data']
		labels[row:row+10000] = (data_dict['labels'])
	return {'data':data, 'labels': labels} 
def load_test(test_path):
	data = np.zeros((10000, 3072))
	labels = np.zeros((10000), dtype=int)
	data_dict = unpickle(test_path)
	data = data_dict['data']
	labels = data_dict['labels']
	return {'data':data, 'labels': labels} 
def split_data(data_dict):
	train_data = {'data': data_dict['data'][:int(0.8*len(data_dict['data']))], 
		'labels': data_dict['labels'][:int(0.8*len(data_dict['labels']))]}
	test_data = {'data': data_dict['data'][int(0.8*len(data_dict['data'])):], 
		'labels': data_dict['labels'][int(0.8*len(data_dict['labels'])):]}
	return train_data, test_data

def Model():
	model = Sequential()
	model.add(Conv2D(32, (3, 3), input_shape=(3, 32, 32), padding='same'))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	
	model.add(Conv2D(32, (3, 3), padding='same'))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Flatten())
	model.add(Dense(512))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(Dense(num_classes, activation='softmax'))
	return model
def Model_2():
	model = Sequential()
	model.add(Conv2D(32, (3, 3), input_shape=(3, 32, 32), padding='same'))
	model.add(BatchNormalization())
	model.add(Activation('relu'))

	model.add(Conv2D(32, (5, 5), input_shape=(3, 32, 32), padding='same'))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	
	model.add(Conv2D(32, (7, 7), padding='same'))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Flatten())
	model.add(Dense(512))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(Dense(num_classes, activation='softmax'))
	return model
def get_activations(model, layer, X_batch):
    get_activations = K.function([model.layers[0].input, K.learning_phase()], model.layers[layer].output)
    activations = get_activations([X_batch,0])
    return activations

if "__name__" != "__main__":
	train_path = sys.argv[1]
	test_path = sys.argv[2]
	all_data = load_data(train_path)
	test_data = load_test(test_path)
	train, val = split_data(all_data)
	X_train = train['data'].astype('float32')
	X_val = val['data'].astype('float32')
	X_train = X_train.reshape(X_train.shape[0], 3, 32, 32)
	X_val = X_val.reshape(X_val.shape[0], 3, 32, 32)

	X_train = X_train / 255.0
	X_val = X_val / 255.0
	y_train = np_utils.to_categorical(train['labels'])
	y_val = np_utils.to_categorical(val['labels'])
	num_classes = y_val.shape[1]
	test = test_data
	X_test = test['data'].astype('float32')
	X_test = X_test.reshape(X_test.shape[0], 3, 32, 32)
	X_test = X_test/255.0
	y_test = np_utils.to_categorical(test['labels'])

	sh = [x for x in range(X_train.shape[0])]
	random.shuffle(sh)
	X_train = np.array([X_train[sh[i],:] for i in sh])
	y_train = np.array([y_train[sh[i]] for i in sh])
	model = Model_2()
	epochs = 25
	lrate = 0.01
	decay = lrate/epochs
	sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)

	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
	model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=32)
	class_dict = {0:'airplane', 1:'automobile',2:'bird',3:'cat',4:'deer',5:'dog',6:'frog',7:'horse',8:'ship',9:'truck'}
																							
	activations_train = get_activations(model, -3, X_train)
	
	activations_test = get_activations(model, -3, X_test)
	
	clf = svm.SVC(decision_function_shape='ovo', kernel='rbf')
	y_train = [row.argmax() for row in y_train]
	#pdb.set_trace()
	y_train = np.array(y_train)
	clf.fit(activations_train[:15000,:], y_train[:15000])
	
	predicted = clf.predict(activations_test)
	y_test = [np.argmax(y_test[i]) for i in range(y_test.shape[0])]
	
	acc = accuracy_score(y_test, predicted)
	with open('q2_b_output.txt', 'w+') as f:
		for i in range(predicted.shape[0]):
			f.write(class_dict[(predicted[i])]+'\n')
	print("Accuracy: %.2f%%" %((acc*100)))