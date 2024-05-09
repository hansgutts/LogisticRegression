import math
import numpy as np
import pandas as pd
import scipy
import scipy.sparse
from utils import *

class logistic() :
    
    def __init__(self, learningRate = 0.01, iterations = 1000, threshold=.5) :
        self.learningRate = learningRate
        self.iterations = iterations
        self.weights = []
        self.bias =0
        self.losses = []
        self.threshold = threshold

    #sigmoid takes in an array of values and puts out an array of sigmoided values
    #sigmoid mathematical function = 1/(1+e^-x)
    def sigmoid(self, x) :
        #perform sigmoid on array. just use x in math since it is numpy array
        return 1/(1+math.e**-x)
    
    #take in an array of predicted and actual values and calculate the loss of our predictions (we dont actually use this anywhere)
    def computeloss(self, pred, actual) :
    def computeloss(self, pred, actual) :

        #total and sum to calculate mean
        total = 0
        sum = 0

        #go through all of our predictions
        for i in range(len(pred)) :

            #increment count of values seen
            total += 1

            #get the predicted value
            yhat = pred[i]
            y = actual[i]

            #yi will always be 1 or 0 so by doing yi*----- and (1-yi)----- 
            #one half will always be 0 and the other 1
            sum += (y * math.log(yhat) - (1-y) * math.log(1-yhat)) 

        #return the loss
        return -sum/total

    #feed_forward is similar to predict. figure out the values of our prediction, except we dont convert to 0 or 1
    #given a vector of vectors we have, just multiply it by the weights and add the bias
    #then put it through sigmoid and return that 
    def feedforward(self, x) :

        #calculate the (weight * input) + bias and return sigmoid of value for probability
        values = (x @ self.weights) + self.bias 
        return self.sigmoid(values)
    
    #fit
    #take input matrix values and array actual
    #each array in values is our feature vector for a specific sample
    #each value in actual is the corresponding class of the instance
    def fit(self, values, actual) :

        #we need to know how many samples we have and the size of our feature vector
        samples, vectorsize = values.shape 

        #initialize bias to 0 and our weights by creating a list of 0's the same size as feature vector
        self.weights = np.zeros(vectorsize) 
        self.bias = 0

        #get transpose of values now to save on processing time when num iterations is large
        valuesT = values.transpose()

        #now we need to loop for the designated number of iterations
        for _ in range(self.iterations) :

            #get the predicted probability values
            A = self.feedforward(values)

            #get how off we are for each prediction
            weightchange = A - actual

            #loop through our weights to update them
            #for each weight, we must look at each correspending input value
            #so for weight theta2 we have 
            #x = [[x1, x2, x3]
            #     [x1, x2, x3]
            #     [x1, x2, x3]]
            #we need to look at every x2 value to update our weight at index 2
            #hence why we declared 
            #xT = [[x1, x1, x1]
            #      [x2, x2, x2]
            #      [x3, x3, x3]]
            #out of this loop
            sum = (valuesT @ weightchange) + self.bias
            
            #update weights based on sum and learning rate
            self.weights -= ((sum*self.learningRate)/samples)

            #the change in bias is just the sum of errors 
            db = np.sum(weightchange)

            #update bias based on db and learning rate
            self.bias -= ((self.learningRate * db)/samples)


    #given a list of samples, return what the predicted values are
    def predict(self, X) :

            #get the sigmoid of each (feature vector * weights) + bias
            result = self.sigmoid((X @ self.weights) + self.bias)

            #since we want classification, return whether or not they are more likely 1 or 0
            return [1 if i > self.threshold else 0 for i in result]