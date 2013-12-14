import numpy as np
import pylab as pl
from sklearn import svm, datasets
from fileio import fread, fwrite

# import some data to play with
#iris = datasets.load_iris()
#X = iris.data[:, :2]  # we only take the first two features. We could                      # avoid this ugly slicing by using a two-dim dataset
#Y = iris.target

def svc(train,test,labels):

	train = np.mat(train)
	test = np.mat(test)	

	# we create an isnwctance of SVM and fit out data. We do not scale our
	# data since we want to plot the support vectors
	C = 1.0  # SVM regularization parameter
	print "Fitting..."
	lin_svc = svm.SVC(kernel='linear', C=C).fit(train, labels)
	print "Predicting..."
	#clf = svm.SVC()
	#clf.fit(train, labels)
	result = lin_svc.predict(test)
	print "Writing..."
	fwrite(result,fname='output.test2.csv')
	#dec = clf.decision_function([[1]])
	#dec.shape[1] # 4 classes: 4*3/2 = 6
	return result



if __name__ == '__main__':
	from fileio import fread, fwrite
	train,labels = fread(f='data/train.csv', train=True)
	test,tmplbl = fread(f='data/test.csv')
	"""train =  train[0:2000]
	labels =  labels[0:2000]
	test =  test[0:2000]
	tmplbl =  tmplbl[0:2000]"""
	result = svc(train, test, labels)
	print "PCA IS: ",result