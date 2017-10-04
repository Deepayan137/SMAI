import pandas as pd 
import numpy as np 
from collections import Counter
import pdb
import sys
path = 'decision_tree_train.csv'
def load_data(path):
	df = pd.read_csv(path)
	X = df.drop('left', axis=1)
	y = df['left']
	return X, y
def load_test_data(path):
	X = pd.read_csv(path)
	return X
def data_split(X, y, fr= 0.8):
	split_index = int(fr*X.shape[0])
	X_train = X.iloc[:split_index,:] 
	y_train = y.iloc[:split_index] 
	X_val = X.iloc[split_index:X.shape[0],:]
	y_val = y.iloc[split_index:X.shape[0]]
	return X_train, y_train, X_val, y_val

def normalize(x):
	avg = x.mean()
	std = x.std()
	return (x-avg)/std

def preprocess_data(X):
	try:
		X['salary'] = X['salary'].map({'low': 0, 'medium': 1, 'high': 2})
		if 'department' in X.columns:
			dept = list(set(X['department']))
			X['department'] = X['department'].map({dept[0]: 0, dept[1]: 1,
				dept[2]: 2, dept[3]: 3, dept[4]: 4, dept[5]: 5, dept[6]: 6,
				dept[7]: 7, dept[8]: 8, dept[9]: 9})
		
		X.loc[X['average_montly_hours'] <= 100, 'average_montly_hours'] = 0
		X.loc[(X['average_montly_hours'] > 100) & (X['average_montly_hours'] <= 150), 'average_montly_hours'] = 1
		X.loc[(X['average_montly_hours'] > 150) & (X['average_montly_hours'] <= 200), 'average_montly_hours'] = 2
		X.loc[(X['average_montly_hours'] > 200) & (X['average_montly_hours'] <= 250), 'average_montly_hours'] = 3
		X.loc[(X['average_montly_hours'] > 250) & (X['average_montly_hours'] <= 300), 'average_montly_hours'] = 4
		X.loc[(X['average_montly_hours'] > 310)] = 5

		X.loc[X['satisfaction_level'] <= 0.25, 'satisfaction_level'] = 0
		X.loc[(X['satisfaction_level'] > 0.25) & (X['satisfaction_level'] <= 0.50), 'satisfaction_level'] = 1
		X.loc[(X['satisfaction_level'] > 0.50) & (X['satisfaction_level'] <= 1), 'satisfaction_level'] = 2
		
		X.loc[X['last_evaluation'] <= 0.40, 'satisfaction_level'] = 0
		X.loc[(X['last_evaluation'] > 0.40) & (X['last_evaluation'] <= 0.60), 'last_evaluation'] = 1
		X.loc[(X['last_evaluation'] > 0.60) & (X['last_evaluation'] <= 0.80), 'last_evaluation'] = 2
		X.loc[(X['last_evaluation'] > 0.80) & (X['last_evaluation'] <= 1), 'last_evaluation'] = 3
		
		return(X)
	except:
		pass
#X, y = load_data(path)

#preprocess_data(X)

def partition(a):
	counter = Counter()
	for item in a:
		counter.update([item])
	return counter

def entropy(a):
	res = 0
	counter = partition(a)
	keys ,vals = zip(*counter.items())
	probs = [(v/float(len(a))) for v in vals]
	
	for p in probs:
		if p != 0.0:
			res -= p * np.log2(p)
	return res

#e = entropy([1,1,1,1,2,2,2,2,2,2,2,2,2,2,2,1,2,2,2,2,2,2,2])
#print(e)

def accuracy_metric(actual, predicted):
	correct = 0
	#pdb.set_trace()
	for i in range(len(actual)):
		if actual.values[i] == predicted[i]:
			correct += 1
	return correct / float(len(actual)) * 100.0

def to_terminal(group):
	#pdb.set_trace()
	result =[]
	count = Counter()
	outcomes = group
	#pdb.set_trace()
	for i in outcomes[0]:
		count.update([i]) 
	
	most_common,num_most_common = count.most_common(1)[0]
	return most_common



def split_data(X, y):
	#pdb.set_trace()
	
	attributes = list(X)
	split_values, group = [],[]
	score, att, v = 999, None, 999
	#print(len(attributes))
	for attribute in attributes:
		#pdb.set_trace()
		#print(attribute)
		#print(set(X[attribute]))
		split_values = [item for item in (set(X[attribute]))]
		#print(len(split_values))
		for value in split_values:
			#print(value)
			left, right = 0, 0
			split_index = value
			left = y.loc[X[attribute] < split_index]
			right = y.loc[X[attribute] >= split_index]
			#pdb.set_trace()
			if len(right) and len(left):
				
				if entropy(left)+entropy(right) < score:			
					 att, v, group =  attribute, value, [[left], [right]]
					#print('Attribute :%s, Value :%f, Score :%f'%(att, v, score))
			else:
				continue
	#pdb.set_trace()
	return{'Attribute': att, 'value':v, 'groups':group} 




def split(X, y, node, max_depth, min_size, depth):
	left, right = node['groups']
	 	
	#pdb.set_trace()
	attribute = node['Attribute']
	#pdb.set_trace()
	del(X[attribute])
	# check for a no split
	if not len(left) or not len(right):
		node['left'] = node['right'] = to_terminal(left + right)
		return
	# check for max depth
	if depth >= max_depth:
		node['left'], node['right'] = to_terminal(left), to_terminal(right)
		return
	# process left child
	if len(left) <= min_size:
		node['left'] = to_terminal(left)
	else:
		node['left'] = split_data(X,y)
		split(X, y, node, max_depth, min_size, depth+1)
	# process right child
	if len(right) <= min_size:
		node['right'] = to_terminal(right)
	else:
		node['right'] = split_data(X,y)
		split(X, y, node, max_depth, min_size, depth+1)

def build_tree(X, y, max_depth, min_size):
	#print(len(list(X)))

	root = split_data(X, y)
	split(X, y, root,max_depth, min_size, 1)
	return root
 
# Make a prediction with a decision tree
def predict(node, X_val, y_val, i):
	#pdb.set_trace()
	if X_val[node['Attribute']].iloc[i] < node['value']:
		if isinstance(node['left'], dict):
			return predict(node, X_val, i)
		else:
			return node['left']
	else:
		if isinstance(node['right'], dict):
			return predict(node['right'], X_val, i)
		else:
			return node['right']

def predict_test(node, X, i):
	if X[node['Attribute']].iloc[i] < node['value']:
		if isinstance(node['left'], dict):
			return predict_test(node, X, i)
		else:
			return node['left']
	else:
		if isinstance(node['right'], dict):
			return predict_test(node['right'], X, i)
		else:
			return node['right']

def decision_tree(X_train, y_train, X_val, y_val, max_depth, min_size):
	tree = build_tree(X_train, y_train, max_depth, min_size)
	predictions = list()
	#pdb.set_trace()
	for i in range(X_val.shape[0]):
		prediction = predict(tree, X_val, y_val, i)

		predictions.append(prediction)

	return(predictions)

def test(X_test, max_depth, min_size):
	tree = build_tree(X_train, y_train, max_depth, min_size)
	predictions = list()
	for i in range(X_test.shape[0]):
		prediction = predict_test(tree, X_test, i)
		sys.stdout.write(str(prediction) + '\n')

if __name__ == '__main__':
	train_path = sys.argv[1]
	test_path = sys.argv[2]
	X, y = load_data(train_path)
	X_train, y_train, X_val, y_val = data_split(X, y)
	X_test = load_test_data(test_path)
	X_train = preprocess_data(X_train)
	X_val = preprocess_data(X_val)
	max_depth = 5
	min_size = 10
	try:
		predictions = decision_tree(X_train, y_train, X_val, y_val, max_depth, min_size)
		accuracy = accuracy_metric(y_val, predictions)
		#print(accuracy)
		test(X_test, max_depth =5, min_size=10)
	except KeyboardInterrupt:
		print("Saving before quit...")
		save(Weights)
    #X_val, y_val = augment_data(X_val, y_val)
    #validation(X_val, y_val, Weights)