import scipy.sparse
from sklearn.model_selection import train_test_split
from sklearn import datasets
from utils import *
import numpy as np
import LogisticRegression
import scipy
import pandas as pd
import pickle

#paths to save pickle files to save on data processing time 
pickleXpath = "XValues.pickle"
pickleYpath = "YValues.pickle"
pickleVocabpath = "UCIDrugVocab.pickle"

#function to make confusion matrix 
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

#beginning logistic regression
print("Working...")

#read in the dataset as a dataframe
print('Opening dataset')
dataset = pd.read_csv("UCIDrugClean.csv")
#dataset = dataset[:30000]

#open our pickled vocabulary
with open(pickleVocabpath, 'rb') as picklefile :
    vocab = pickle.load(picklefile)

#create our vocab location dictionary
#need this to create consistent bag of word vectors
print("Creating vocab information")
x = 0
vocabLocationDictionary = {}
vocabSize = len(vocab)
for word in vocab:
    vocabLocationDictionary[word] = x
    x += 1 


print("Creating bow vectors")

#create bag of words for each review
#since we are making sparse matrix we need to collect the value and the value's corresponding row and col
#the row is our i as it incrememnts every new vector
#the col is the location of the word in the bag of word vector as defined above
#the data is just a 1 for now (occurence seems more important than count)
i=0
data = []
row = []
col = []
for entries in dataset.itertuples() :
    review = entries.review
    for word in review.split() :
        if word in vocabLocationDictionary :
            data.append(1)
            row.append(i)
            col.append(vocabLocationDictionary[word])

    i += 1
    
#finally, convert the reviews into our bag of words matrix
X = scipy.sparse.csr_matrix((data, (row, col)), shape = (len(dataset), vocabSize))
y = list(dataset['rating'])

'''#pickle the bow and class vectors to save on processing time
with open(pickleXpath, 'wb') as pickleFile : #pickle logic/info from https://stackoverflow.com/questions/11218477/how-can-i-use-pickle-to-save-a-dict-or-any-other-python-object
    pickle.dump(X, pickleFile, pickle.HIGHEST_PROTOCOL)

with open(pickleYpath, 'wb') as pickleFile : 
    pickle.dump(y, pickleFile, pickle.HIGHEST_PROTOCOL)'''

#open pickled files
with open(pickleXpath, 'rb') as picklefile :
    X = pickle.load(picklefile)

with open(pickleYpath, 'rb') as picklefile :
    y = pickle.load(picklefile)

#split into training and testing data
X_train = X[:round(X.shape[0]*.8)]
y_train = y[:round(len(y)*.8)]
X_test = X[round(X.shape[0]*.8):]
y_test = y[round(len(y)*.8):]

'''#snipping for testing logistic regression, I know the expected result
dataset = datasets.load_breast_cancer()
X, y = dataset.data, dataset.target

X, y = dataset.data, dataset.target 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1234
)'''

#try to find the right learning rate and number of iterations
acc = []
#for x in range(5, 1000, 5) :
#    for y in range(1, 1000, 5) :
        #train the logistic regression
print("Training logistic regression...")
regressor = LogisticRegression.logistic(learningRate=1, iterations=100)
regressor.fit(X_train, y_train)

#test the logistic regression
print("Testing logistic regression")
predictions = regressor.predict(X_test)
#test the logistic regression
print("Testing logistic regression")
predictions = regressor.predict(X_test)

#evaluate the logistic regression
cm ,accuracy,sens,precision,f_score  = confusion_matrix(np.asarray(y_test), np.asarray(predictions))
#acc.append([x, y, accuracy])
print("Test accuracy: {0:.3f}".format(accuracy))
print("Confusion Matrix:\n", np.array(cm))
#evaluate the logistic regression
cm ,accuracy,sens,precision,f_score  = confusion_matrix(np.asarray(y_test), np.asarray(predictions))
#acc.append([x, y, accuracy])
print("Test accuracy: {0:.3f}".format(accuracy))
print("Confusion Matrix:\n", np.array(cm))

#sort our accuracies in descending order and display them
#acc.sort(key=lambda x: x[2])
#print(acc)