#!/usr/bin/python
#import csv_io
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score
from numpy import genfromtxt, savetxt
from preproc import pca

def randforest(train,test,labels, estimators=800, runMedian=True, runPCA=True, components=800):
	train = np.mat(train)
	test = np.mat(test)
	print "Random Forest Classifier\n"
	print "Reading in data-sets..."
	#train = csv_io.read_data("train.csv")
	#test = csv_io.read_data("test.csv")
	#Required util function, found built in func to do it for us.
	#train = genfromtxt(open('data/train.csv', 'r'), delimiter=',', dtype='int64')[1:]
	#test = genfromtxt(open('data/test.csv', 'r'), delimiter=',', dtype='int64')[1:]
	#labels = [x[0] for x in train]
	#train = [x[1:] for x in train]

	print "Starting Classifier"
	tree = RandomForestClassifier(n_estimators=800, n_jobs=-1)

	if (runPCA==True):
			trainReduce, testReduce = pca(train,test,components=600)
			tree.fit(trainReduce,labels)	#print at beginning of this
			result = tree.predict(testReduce)
			print "Writing output to file output.knn-kdtree-pca.csv\n"
			fwrite(result,fname='output.rf-pca.csv')

	print "Running without PCA..."
	tree.fit(train,labels)
	
	print "Check Model against test data"
	probability = tree.predict_proba(test)
	#Don't forget this is a predicted probability
	probability = ["%f" % x[1] for x in probability]
	print "probability is: ",probability
	result = tree.predict(test)
 	print "Writing output to 'output.rf.csv'"
	fwrite(result,fname='output.rf.csv' )

if __name__ == '__main__':
	from fileio import fread, fwrite
	train,labels = fread(f='data/train.csv', train=True)
	test,tmplbl = fread(f='data/test.csv')
	randforest(train,test,labels, runPCA=True, components=800)


		