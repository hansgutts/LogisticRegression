import math
import numpy as np
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
        return [1/(1+(math.e**-num)) for num in x]
    
    #take in an array of predicted and actual values and calculate the loss of our predictions
    def computeLoss(self, pred, actual) :

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
        values = []
        for vector in x :
            value = 0
            for i in range(len(vector)) :
                value += vector[i] * self.weights[i]
            value += self.bias
            values.append(value)
        return self.sigmoid(values)
    
    #fit
    #get the number of entries and size of input vector
    #size of input vector is your weights size
    #(dont forget to actually initialize your vectors)
        #they are doing matrix multiplaction (in feed forward) to get array of outputs
        #   we could just loop? maybe worse but I get it better
        #gradient descent
    def fit(self, values, actual) :

        #get dimensions of our input vector
        #it will be multidimensional
        samples = len(values) #samples is the number of training documents
        vectorsize = len(values[0]) #vectorsize is how big our actual input vector is for classification

        #initialize our weights. multiplying vector gets us full 0 vector
        self.weights = [0] * vectorsize
        self.bias = 0

        #now we need to loop for iterations
        for _ in range(self.iterations) :
            
            #get the predicted values
            A = self.feedforward(values)

            #get how off we are for each prediction
            weightchange = [a-y for (a, y) in zip(A, actual)]

            #loop through our weights to update them
            for weightindex in range(len(self.weights)) :
                sum = 0
                count = 0 

                #for each weight, we must look at each correspending input value
                #so for weight theta2 we have 
                #x = [[x1, x2, x3]
                #     [x1, x2, x3]
                #     [x1, x2, x3]]
                #we need to look at every x2 value to update our weight at index 2
                for i in range(len(weightchange)) :
                    sum += weightchange[i]*values[i][weightindex]
                    count += 1
                
                self.weights[weightindex] -= (self.learningRate*sum)/count
            db = 0
            for i in range(len(weightchange)) :
                db += weightchange[i]
            biaschange = self.learningRate * db * (1/count)
            self.bias -= biaschange


    def predict(self, X) :
            result = []
            for vector in X :
                output = 0
                for i in range(len(vector)) :
                    output += self.weights[i] * vector[i]
                output += self.bias
                result.append(output)
            result = self.sigmoid(result)
            return [1 if i > self.threshold else 0 for i in result]

















'''class logistic() :
    
    def __init__(self, learningRate = 0.01, iterations = 1000, threshold=.5, weights = None, bias = None) :
        self.learningRate = learningRate
        self.iterations = iterations
        self.weights = weights
        self.bias = bias
        self.losses = []
        self.threshold = threshold

    def sigmoid(self, x) :
        return [1/(1+(math.e**-num)) for num in x]
    
    def computeLoss(self, actual, pred) :
        total = 0
        sum = 0
        for i in range(len(pred)) :
            total += 1
            pi = pred[i]
            yi = actual[i]
            #yi will always be 1 or 0 so by doing yi*----- and (1-yi)----- 
            #one half will always be 0 and the other 1
            sum += (yi * math.log(pi) - (1-yi) * math.log(1-pi)) 

        return -sum/total

    #feed_forward
    #given a vector of vectors we have, just multiply it by the weights and add the bias
    #then put it through sigmoid and return that 
    def feedforward(self, x) :
        values = []
        for vector in x :
            value = 0
            for i in range(len(vector)) :
                value += vector[i] * self.weights[i]
            value += self.bias
            values.append(value)
        return self.sigmoid(values)
    
    #fit
    #get the number of entries and size of input vector
    #size of input vector is your weights size
    #(dont forget to actually initialize your vectors)
        #they are doing matrix multiplaction (in feed forward) to get array of outputs
        #   we could just loop? maybe worse but I get it better
        #gradient descent
    def fit(self, values, actual) :
        samples = len(values)
        vectorsize = len(values[0])

        self.weights = [0] * vectorsize
        self.bias = 0

        #now we need to loop for iterations
        for _ in range(self.iterations) :
            
            A = self.feedforward(values)
            weightchange = [a-y for (a, y) in zip(A, actual)]
            for weightindex in range(len(self.weights)) :
                sum = 0
                count = 0 
                for i in range(len(weightchange)) :
                    sum += weightchange[i]*values[i][weightindex]
                    count += 1
                
                self.weights[weightindex] -= (self.learningRate*sum)/count
            db = 0
            for i in range(len(weightchange)) :
                db += weightchange[i]
            biaschange = self.learningRate * db * (1/count)
            self.bias -= biaschange
    def predict(self, X) :
            
            result = []
            for vector in X :
                output = 0
                for i in range(len(vector)) :
                    output += self.weights[i] * vector[i]
                output += self.bias
                result.append(output)
            print(result)
            result = self.sigmoid(result) 
            print(result)          
            return [1 if i > self.threshold else 0 for i in result]

'''