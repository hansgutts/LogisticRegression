import scipy.sparse
import scipy.sparse
from sklearn.model_selection import train_test_split
from sklearn import datasets
from utils import *
import numpy as np
import LogisticRegression
import math
import scipy
import math
import scipy

import pandas as pd
import pickle

def confusion_matrix(y_actual, y_predicted):
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    epsilon = 1e-9
    for i in range(len(y_actual)):
        if y_actual[i] > 0:
            if y_actual[i] == y_predicted[i]:
                tp = tp + 1
            else:
                fn = fn + 1
        if y_actual[i] < 1:
            if y_actual[i] == y_predicted[i]:
                tn = tn + 1
            else:
                fp = fp + 1

    cm = [[tn, fp], [fn, tp]]
    accuracy = (tp+tn)/(tp+tn+fp+fn+epsilon)
    sens = tp/(tp+fn+epsilon)
    prec = tp/(tp+fp+epsilon)
    f_score = (2*prec*sens)/(prec+sens+epsilon)
    return cm,accuracy,sens,prec,f_score

print("Working...")

print('Opening dataset')
dataset = pd.read_csv("UCIDrugClean.csv")
dataset = dataset[:20000]

#train, test = train_test_split(dataset, test_size=0.2)

with open("UCIDrugVocab.pickle", 'rb') as picklefile :
    vocab = pickle.load(picklefile)


print("Creating vocab information")
x = 0
vocabLocationDictionary = {}
vocabSize = len(vocab)
for word in vocab:
    vocabLocationDictionary[word] = x
    x += 1 


print("Creating bow vectors")
X = []

for entries in dataset.itertuples() :
    review = entries.review
    newx = [0] * vocabSize

    for word in review.split() :
        if word in vocabLocationDictionary :
            newx[vocabLocationDictionary[word]] = 1

    X.append(newx)
    #X.append(scipy.sparse.coo_array(newx))

#dataset['X'] = pd.Series(X)
X = scipy.sparse.csr_matrix(X)
y = list(dataset['rating'])

X_train = X[:round(X.shape[0]*.8)]
y_train = y[:round(len(y)*.8)]
X_test = X[round(X.shape[0]*.8):]
y_test = y[round(len(y)*.8):]

#print(f"X train {len(X_train)} X test {len(X_test)} y test {len(y_test)}")




'''dataset = datasets.load_breast_cancer()
X, y = dataset.data, dataset.target

X, y = dataset.data, dataset.target 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1234
)'''

print("Training logistic regression...")
regressor = LogisticRegression.logistic(learningRate=1, iterations=1)
regressor.fit(X_train, y_train)
print(regressor.weights)
print(regressor.bias)

print("Testing logistic regression")
predictions = regressor.predict(X_test)
#print(predictions)
#print(y_train)
cm ,accuracy,sens,precision,f_score  = confusion_matrix(np.asarray(y_test), np.asarray(predictions))
print("Test accuracy: {0:.3f}".format(accuracy))
print("Confusion Matrix:", np.array(cm))
print("Confusion Matrix:", np.array(cm))


'''print(np.dot([[1, 1, 2, 1], [2, 2, 2, 2]], [2, 2, 2, 2]))
x=np.ndarray([1, 2, 3, 4])

print(1 / (1 + np.exp(-x)))'''