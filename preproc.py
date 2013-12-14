#!/usr/bin/python
import numpy as np 
from sklearn.neighbors import KNeighborsClassifier
from sklearn import decomposition
from scipy import ndimage

def pca(train,test=None,components=800):
	print "Running PCA with %d components..." %components
	decomp = decomposition.PCA(n_components=components).fit(train)
	if (test == None):
		trainReduce = decomp.transform(train)
		return train
	else:
		trainReduce = decomp.transform(train)
		testReduce = decomp.transform(test)
		return trainReduce, testReduce

def medianfilter(train):
	#train = np.matrix(train)

	#imageM = np.matrix(imageM)
	#imageM = imageM.reshape(28,28)
	#count = 0
	#len(train)
	for k in range (0,len(train)):
		#start=k*784
		#end = (k+1)*784
		#print "Train is type: ",type(train)

		tempImgArray = np.array(train[k][:])
		imageM = tempImgArray.reshape(28,28)
		#print "ImageM type is: ",type(imageM)
		#print "ImageM: ",imageM
		med_denoised = ndimage.median_filter(imageM, size(2,2))
		#print "denoised type before: ",type(med_denoised), len(med_denoised)
		med_denoised_flat = med_denoised.reshape(1,784)
	#	print "denoised type after: ",type(med_denoised_flat), len(med_denoised_flat)
		#print "med_denoised_flat",med_denoised_flat
		train[k][:] = med_denoised_flat

		#print "train[k]: ", train[k][:]
		#fwrite(train,fname="output.median.csv")

	return train


if __name__ == '__main__':
	from fileio import fread, fwrite
	train,labels = fread(f='data/train.csv', train=True)
	test,tmplbl = fread(f='data/test.csv')
	#pca(train, test)
	medianfilter(train,test)