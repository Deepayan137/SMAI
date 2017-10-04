# -*- coding: utf-8 -*-
import sys
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.metrics import r2_score
import pdb

def lasso_regr(X_train, Y_train, alpha):
	regr = linear_model.Lasso(alpha=alpha)
	regr.fit(X_train, Y_train)
	return regr

def accuracy(X_test, Y_test, regr_model):
	predicted = regr_model.predict(X_test)
	predicted = list(map(lambda x: 1 if x >0.5 else 0, predicted))
	acc = sum((list(map(lambda x, y: 1 if x==y else 0,predicted,Y_test))))/float(len(predicted))
	
	return acc, predicted

def display_prediction(predictions):
	for i in range(len(predictions)):
		sys.stdout.write(str(predictions[i]) + '\n')

if __name__ == '__main__':
	train = sys.argv[1]
	test  = sys.argv[2]
	df_train = pd.read_csv(train, header=None)
	df_test = pd.read_csv(test, header=None)
	X_train = np.array(df_train.iloc[:,:-1])
	Y_train = np.array(df_train.iloc[:,-1])
	X_test = np.array(df_test.iloc[:,:-1])
	Y_test = np.array(df_test.iloc[:,-1])
	regr_model = lasso_regr(X_train, Y_train, 0.1)
	accuracy, predictions = accuracy(X_test, Y_test, regr_model)
	display_prediction(predictions)
	#print(accuracy)	
