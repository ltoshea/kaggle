#!/usr/bin/python
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from preproc import pca,medianfilter

def knn(train,test,labels,neighbours=10, runAll=None, median=True, runPCA=True, components=80):
	print "Putting training data into matrix"
	trainM = np.mat(train)
	print "Running K nearest Neighbour, default = 10"
	knn = KNeighborsClassifier(n_neighbors=neighbours, algorithm="kd_tree")


	if(median == True):
		print "Running data through Median filter..."
		trainfilter = medianfilter(train)
		knn.fit(trainfilter,labels)
		result = knn.predict(trainfilter)
		print "Writing to output file output.knn-kdtree-median.csv\n"
		fwrite(result,fname='output.knn-kdtree-median.csv')
		return(0)

	if (runPCA==True):
			trainReduce, testReduce = pca(train,test,components)
			knn.fit(trainReduce,labels)	#print at beginning of this
			result = knn.predict(testReduce)
			print "Writing output to file output.knn-kdtree-pca.csv\n"
			fwrite(result,fname='output.knn-kdtree-pca.csv')
	


	
	print "Running without PCA\n"
	knn.fit(trainM,labels) #need this here - might as well print
	result = knn.predict(test)
	print "Writing output to file output.knn-kdtree.csv\n"
	fwrite(result,fname="output.knn-kdtree.csv")
	#return result



if __name__ == '__main__':
	from fileio import fread, fwrite
	train,labels = fread(f='data/train.csv', train=True)
	test,tmplbl = fread(f='data/test.csv')
	result = knn(train, test, labels,runPCA=False)
	#print "RESULT IS: ",result
