#!/usr/bin/env python
import sys
import os
import re
import pdb
import time
import pickle
import random
import numpy as np
import shutil
from collections import Counter
import codecs
import pandas as pd
import math
import time
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, confusion_matrix


class FeatureVector(object):
	def __init__(self,vocab,numdata):
		self.vocab = vocab
		self.vocabsize = len(self.vocab)
		self.X =  np.zeros((numdata,self.vocabsize), dtype=np.int)
		self.Y =  np.zeros((numdata,), dtype=np.int)

	def make_featurevector(self, input, classid, idx):
		
		self.Y[idx] = classid
		word_count = Counter()
		for each_input in inputs:
			word_count.update([each_input])
		
		for item in word_count:
			try:
				pos = self.vocab.index(item)
				self.X[idx, pos] = word_count[item]
			except ValueError:
				pass

class KNN(object):
	def __init__(self,trainVec,testVec):
		self.X_train = trainVec.X
		self.Y_train = trainVec.Y
		self.X_test = testVec.X
		self.Y_test = testVec.Y
		self.metric = Metrics('accuracy')
		
	def shuffle_train(self, X,y):
		s = [i for i in range(X.shape[0])]
		X_shuffled = np.zeros((X.shape[0], X.shape[1]), dtype=np.int)
		y_shuffled = np.zeros(len(y), dtype=np.int)
		random.shuffle(s)
		for i in range((X.shape[0])):
			X_shuffled[i,:] = X[s[i],:]
			y_shuffled[i] = y[s[i]]
		return X_shuffled, y_shuffled
	def classify(self, nn):
		
		acc, f1 = 0.0, 0.0
		dist_mat = self.distance_matrix(self.X_train, self.X_test)
		dist_mat = dist_mat/float(self.X_train.shape[1])
		X_smallest=[]
		for j in range(dist_mat.shape[1]):
			X_smallest.append(np.argsort(dist_mat[:,j])[:nn])
		pred = [self.vote(X_smallest[i]) for i in range(len(X_smallest))]
		self.test(pred)
		acc = (self.metric.accuracy(pred, self.Y_test))
		f1 = (self.metric.f1_score(pred, self.Y_test, nn))
		conf_mat = self.metric.get_confmatrix(pred, self.Y_test, nn)
	
		return acc, f1
	def test(self, predictions):
		classes = ['galsworthy','galsworthy_2','mill','shelley','thackerey','thackerey_2','wordsmith_prose','cia','johnfranklinjameson','diplomaticcorr']
		for prediction in predictions:
			sys.stdout.write(classes[prediction] + '\n')

	def euclideanDistance(self,instance1, instance2):
		distance = 0
		for x in range(len(instance1)):
			distance += pow((instance1[x] - instance2[x]), 2)
		return math.sqrt(distance)

	def distance_matrix(self, train, test):
		dist_mat = np.zeros((train.shape[0], test.shape[0]), dtype = np.int)
		for i in range(1,train.shape[0]):
			for j in range(1,test.shape[0]):
				data1 = train[i,:]
				data2 = test[j,:]
				dist_mat[i,j] = np.sqrt(np.sum((train[i,:] - test[j,:])**2))
		return dist_mat

	def vote(self, X_s):
		
		max_counter = Counter()
		for unit in X_s:
			max_counter.update([self.Y_train[unit]])
		most_common,num_most_common = max_counter.most_common(1)[0]
		return most_common

class Metrics(object):
	def __init__(self,metric):
		self.metric = metric

	def score(self):
		if self.metric == 'accuracy':
			return self.accuracy()
		elif self.metric == 'f1':
			return self.f1_score() 
	def get_confmatrix(self,y_pred,y_test, nn):
		m = np.zeros((10, 10), dtype = int)
		for pred, exp in zip(y_pred, y_test):
			m[pred-1][exp-1] += 1
		df  = pd.DataFrame(m)
		df.to_csv('confusion_mat_%d.csv'%nn)
		return m
	def accuracy(self,y_pred, y_test):
		correct = np.sum(y_pred == y_test)
		accuracy = (correct/float(len(y_test)))*100
		return accuracy
	def f1_score(self, y_pred, y_test, nn):
		
		conf_mat = self.get_confmatrix(y_pred, y_test, nn)
		tp, fp, tn, precison, recall, f_score  = 0, 0, 0, 0.0, 0.0, 0.0
		rows, cols = conf_mat.shape[0], conf_mat.shape[1]
		for i in range(rows):

			fp += sum(conf_mat[i, :])- conf_mat[i, i]
			tp += conf_mat[i, i]
			tn += sum(conf_mat[:, i])- conf_mat[i, i]
		
		precison = tp/float(fp+tp)
		recall = tp/float(tp+tn)
		
		f_score = 2*(precison*recall)/(precison+recall)
		
		return f_score

def get_words_pagewise(file_path):
	try:
		words=[]

		with codecs.open(file_path, "r",encoding='utf-8', errors='ignore') as text:
			lines = text.readlines()
			f =  lambda x: re.findall(r'<s>(.*)<\\s>', x)
			for line in lines:
				try:
					words.append(f(line)[0].split())
				
				except IndexError:
					words.append(line.split())
					continue
			words = [item for sublist in words for item in sublist]
			#words = [word for word in words if word not in stop_words]
			words = [word for word in words if 7<len(word)<15]
			return words
	except Exception as e:
		print (e)
		

def vocabulary(path):
	all_words =[]
	for dr, drs, fls in os.walk(os.path.join(path)):
		for each_file in fls:
			file_path = os.path.join(dr,each_file)
			#print(file_path)
			words = get_words_pagewise(file_path)
			all_words.extend(words)
	return list(set(all_words))



def split_data(my_class):
	class_path = os.path.join(datadir,my_class)
	files = os.listdir(class_path)
	n_files  = len(os.listdir(class_path))
	train_path = os.path.join(datadir,'train',my_class)
	test_path = os.path.join(datadir,'test',my_class)
	if os.path.exists(train_path) != True:
		os.mkdir(train_path)
	if os.path.exists(test_path) != True:
		os.mkdir(test_path)
	list1 = list(range(n_files))
	train_fr = int(0.8 * n_files)
	for i in range(0, train_fr):
		shutil.copy(os.path.join(class_path,files[i]), train_path)

	for i in range(train_fr, n_files):
		shutil.copy(os.path.join(class_path,files[i]), test_path)
	#print('data splitting completed...')

if __name__ == '__main__':
	start_time = time.time()
	k_neighbours = [3, 5, 7, 9, 11]
	datadir = 'data/'
	classes = ['galsworthy/','galsworthy_2/','mill/','shelley/','thackerey/','thackerey_2/','wordsmith_prose/','cia/','johnfranklinjameson/','diplomaticcorr/']
	train = sys.argv[1]
	test = sys.argv[2]
	inputdir = [train, test]
	if os.path.exists('vocabulary.pickle') != True:
		#print('building vocabulary...')
		vocab = vocabulary(inputdir[0])
		with open('vocabulary.pickle', 'wb') as f:
			pickle.dump(vocab, f, protocol=2)
	else:
		#print('loading the vocabulary file ...')
		with open('vocabulary.pickle', 'rb') as f:
			vocab = pickle.load(f)
	
	vocab.sort(key = lambda x:len(x))
	vocab = vocab[-20000:]
	train_dir = [os.path.join(inputdir[0], item) for item in os.listdir(os.path.join(inputdir[0]))]
	test_dir = [os.path.join(inputdir[1], item) for item in os.listdir(os.path.join(inputdir[1]))]
	f = lambda x: len(os.listdir(x))
	trainsz = sum(list(map(lambda x: f(x), train_dir)))
	testsz = sum(list(map(lambda x: f(x), test_dir)))
	
	if os.path.exists('trainVec.pickle') != True:
		
		trainVec = FeatureVector(vocab,trainsz)
		testVec = FeatureVector(vocab,testsz)
		all_words =[]
		tr, te = 0, 0
		count_train, count_test =0,0

		for idir in inputdir:
			classid = 0
			for c in classes:
				listing = os.listdir(idir+c)
				random.shuffle(listing)
				for filename in listing:
					f = (idir+c+filename)
					inputs = get_words_pagewise(f)
					
					inputs.sort(key = lambda x:len(x))
					if idir == 'data/train/':
						trainVec.make_featurevector(inputs[-300:], classid, tr)
						tr+=1
						count_train+=1
						#print(count_train)

					else:
						testVec.make_featurevector(inputs[-300:], classid, te)
						te += 1
						count_test+=1
						#print(count_test)
				classid += 1

		
		with open('trainVec.pickle', 'wb') as train_f:
			pickle.dump(trainVec, train_f, protocol=2)
		with open('testVec.pickle', 'wb') as test_f:
			pickle.dump(testVec, test_f, protocol=2)
	else:
		
		with open('trainVec.pickle', 'rb') as train_f:
			trainVec = pickle.load(train_f)

		with open('testVec.pickle', 'rb') as test_f:
			testVec = pickle.load(test_f)
	knn = KNN(trainVec,testVec)
	knn.classify(3)
