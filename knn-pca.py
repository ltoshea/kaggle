#!/usr/bin/python
import numpy as np 
from sklearn.neighbors import KNeighborsClassifier
from sklearn import decomposition
from preproc import pca

def knn_pca(train,test,labels,components=80,neighbours=10):
	print "Putting training data into matrix"
	train = np.mat(train)	
	knn = KNeighborsClassifier(n_neighbors=neighbours, algorithm="kd_tree")
	
	trainReduce, testReduce = pca(train,test,components)
	print knn.fit(trainReduce,labels)	
	result = knn.predict(testReduce)
	print "Writing output to file 'output.knn-pca.csv'"
	fwrite(result,fname='output.knn-pca.csv')
	#return result



if __name__ == '__main__':
	from fileio import fread, fwrite
	train,labels = fread(f='data/train.csv', train=True)
	test,tmplbl = fread(f='data/test.csv')
	knn_pca(train, test, labels)
	#result = knn_pca(train, test, labels)
	#print "PCA IS: ",result
