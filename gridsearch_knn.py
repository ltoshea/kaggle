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
from sklearn.neighbors import KNeighborsClassifier
import time
print(__doc__)

# Loading the Digits dataset
start = time.clock()
# To apply an classifier on this data, we need to flatten the image, to
# turn the data in a (samples, feature) matrix:
#X = digits.images.reshape((n_samples, -1))
#y = digits.target

train,labels = fread(f='data/train.csv', train=True)
train = medianfilter(train)
train = pca(train)

#print "lenth 800 1000 1200: ", len(train800), " | " , len(train1000) , " | " , len(train1200)

#tuned_parameters = [{'kernel': ['rbf'], 'gamma': [10, 1, 1e-3, 1e-4], 'C': [1, 10, 100, 1000]}]
#tuned_parameters = [{'n_estimators': [10,15,20], 'components': [800, 1000, 1200]}]
#tuned_parameters = [{'algorithm': ['kd_tree'], 'n_neighbors': [10, 50, 100, 200], 'leaf_size' : [30, 60, 150]}] 
tuned_parameters = [{'algorithm': ['ball_tree'], 'n_neighbors': [10, 50, 100, 200], 'leaf_size' : [30, 60, 150], 'degree' : [2,3,4]}] 

max_features = sqrt(784)
scores = ['precision', 'recall']
#n_samples = len(X)/4
# Split the dataset in two equal parts

X_train, X_test, y_train, y_test = train_test_split( train, labels, test_size=0.5, random_state=0)

for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()

    #clf = GridSearchCV(SVC(C=1), tuned_parameters, cv=5, scoring=score)
    #clf = GridSearchCV(SVC(cache_size=800, probability=True, shrinking=True, tol=0.001), tuned_parameters, cv=10, scoring=score)
    clf = GridSearchCV(KNeighborsClassifier(weights='distance', p=2, metric='minkowski'), tuned_parameters, cv=10, scoring = score)
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
    #blockelapsed = (time.clock() - start)
    #print ("time elapsed: " %blockelapsed)
elapsed = (time.clock() - start)
print ("time elapsed: ",elapsed)