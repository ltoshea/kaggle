from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import cross_validation, svm
from fileio import fread, fwrite
from preproc import medianfilter
from numpy import genfromtxt
import numpy as np
import time


def main():
    #read in  data, parse into training and target sets
    data = genfromtxt(open('data/train.csv', 'r'), delimiter=',', dtype='int64')[1:]

    target = np.array( [x[0] for x in data] )
    train = np.array( [x[1:] for x in data] )
    #train = medianfilter(train)
   # print len(target)

    #In this case we'll use a random forest, but this could be any classifier
    #cfr = RandomForestClassifier(n_estimators=100)
    start = time.clock()
    #cfr = svm.SVC(kernel='poly',C=1.0, degree=1)
    #cfr = svm.SVC(C=10.0, cache_size=200, class_weight=None, degree=3, gamma=0.1, kernel='rbf', max_iter=-1, probability=True, verbose=False)
    cfr = svm.SVC(C=1000000.0, cache_size=200, class_weight=None, coef0=0.0, degree=3, gamma=0.0001, kernel='rbf', max_iter=-1, probability=False,shrinking=True, tol=0.001, verbose=False)
    print cfr
    #cfr = KNeighborsClassifier(n_neighbors=neighbours, algorithm="kd_tree")


    #Simple K-Fold cross validation. 10 folds.
    cv = cross_validation.KFold(len(train), n_folds=10, indices=True)
 
   #iterate through the training and test cross validation segments and
    #run the classifier on each one, aggregating the results into a list
    results = []
    i = 0
    count =0
    for traincv, testcv in cv:
        ClassPred = cfr.fit(train[traincv], target[traincv]).predict(train[testcv]) 
        for j in range(0,(len(train)/10)):
         #   print "prediction: ", probas[j], "  target: ", target[j]
            labelindex = testcv[j]
            if (ClassPred[j] == target[labelindex]):
                i = i+1
        #print "i", i
        accuracy = (i/2000.0)*100
        i=0 
        results.append(accuracy)

        count = count + 1
        elapsedfold = (time.clock() - start)
        print "accuracy for fold", count, " : ", accuracy,"%"
        print "time after fold", count, " : ", elspasedfold,"%"
        #print "probas length: ", len(probas)
        #print "probas: ", probas
       # cfr.cross_validation.cross_val_score(cfr, train[traincv], y=None, scoring=None, cv=None, n_jobs=1, verbose=0, fit_params=None, score_func=None, pre_dispatch='2*n_jobs')

    #print out the mean of the cross-validated results
    elapsed = (time.clock() - start)
    
    print "Results for RBF, c = 10.0, gamma=0.1 \n" + str(np.array(results).mean())
    print "Time taken is %ds" % elapsed


if __name__=="__main__":
    main()