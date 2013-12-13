from __future__ import print_function

from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from fileio import fread
import time
print(__doc__)

# Loading the Digits dataset
start = time.clock()
# To apply an classifier on this data, we need to flatten the image, to
# turn the data in a (samples, feature) matrix:
#X = digits.images.reshape((n_samples, -1))
#y = digits.target

X,Y = fread(f='data/train.csv', train=True)
#n_samples = len(X)/4
#print ("n_samples is: ",n_samples)
max_features = sqrt(784)
# Split the dataset in two equal parts
X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size=0.5, random_state=0)

# Set the parameters by cross-validation
#tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4], 'C': [1, 10, 100, 1000]}, {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
tuned_parameters = [{'estimators': [10,15,20], 'components': [800, 1000, 1200]}

scores = ['precision', 'recall']

for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()

    #clf = GridSearchCV(SVC(C=1), tuned_parameters, cv=5, scoring=score)
    clf = GridSearchCV(randforest(estimators=800, runPCA=True, components=800):
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
    print ("time elapsed: " %block_elapsed)
elapsed = (time.clock() - start)
print ("time elapsed: ",elapsed)
# Note the problem is too easy: the hyperparameter plateau is too flat and the
# output model is the same for precision and recall with ties in quality.