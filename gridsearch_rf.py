from __future__ import print_function

from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from fileio import fread
from preproc import pca,medianfilter
from math import sqrt
from sklearn.ensemble import RandomForestClassifier
import time
import sys,os
print(__doc__)

# Loading the Digits dataset
start = time.clock()
# To apply an classifier on this data, we need to flatten the image, to
# turn the data in a (samples, feature) matrix:
#X = digits.images.reshape((n_samples, -1))
#y = digits.target

train,labels = fread(f='data/train.csv', train=True)
train = medianfilter(train)
train20 = pca(train,components=20)
train30 = pca(train,components=30)
train60 = pca(train,components=60)


#tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4], 'C': [1, 10, 100, 1000]}, {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
tuned_parameters = [{'n_estimators': [10,15,20,30,50,100]}]
max_features = sqrt(784)
scores = ['precision', 'recall']
#n_samples = len(X)/4
# Split the dataset in two equal parts

#//,train1200
for i in range(0,3):
    """print ("i is: ",i[0:1])
    #time.sleep(10)
    print ("i is: ",i[1:2])
    #time.sleep(10)
    print ("train800",train800[0:1])
    #time.sleep(10)
    print ("train800",train800[1:2])
    print ("end of for")
    continue"""
    if (i==0):
        train = train20
        print ("\n Running with PCA:800 \n")
    if (i==1):
        train = train30
        print ("Running with PCA:1000 \n")
    if (i==2):
        train = train60
        print ("Running with PCA:1200 \n")


    X_train, X_test, y_train, y_test = train_test_split( train, labels, test_size=0.1, random_state=0)


    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        print()

        #clf = GridSearchCV(SVC(C=1), tuned_parameters, cv=5, scoring=score)
        clf = GridSearchCV(RandomForestClassifier(max_features=max_features, n_jobs=-1), tuned_parameters, cv=10)
        clf.fit(X_train, y_train)

        print("Best parameters set found on development set:")
        print()
        print(clf.best_estimator_)
        print()
        print("Grid scores on development set:")
        print()
        for params, mean_score, scores in clf.grid_scores_:
            print("%0.3f (+/-%0.03f) for %r"
                  % (mean_score, scores.std() / 2, params))
        print()

        print("Detailed classification report:")
        print()
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
        print()
        y_true, y_pred = y_test, clf.predict(X_test)
        print(classification_report(y_true, y_pred))
        print()
        block_elapsed = (time.clock() - start)
        print ("time elapsed: ",str(block_elapsed))
    elapsed = time.clock() - start
    print ("time elapsed: ",elapsed)