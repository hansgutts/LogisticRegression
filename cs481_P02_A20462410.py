import sys
import csv
import math
import json

path = "UCIdrugClean.csv"
trainsize = 80

#I want to use logistic regression along with a bag of words approach to perform sentiment analysis
#we will need a vocabulary, which I have created in UCIDrugVocab.py



#take in the data, split it, and get the bag of words vector and output probability vectors for positive/negative
def main() :

    print("Training classifier...")

    #open the csv file (errors='ignore' because there was a strange ascii error)
    with open(path, errors='ignore') as data :

        #reader for data
        datareader = csv.reader(data)

        #list to hold the entries we will be training/testing
        datalist = []
        i = 0

        #get data into a list (easier to manipulate, maybe inneficient)
        for line in datareader :
            datalist.append(line)

        #calculate number of entries to test on based on input value
        trainnum = int(round(len(datalist) * trainsize, 0))

        #split data in training and testing set based on input value
        train = datalist[0:trainnum]
        test = datalist[trainnum:]

        #where to actually train and get the "weights"
        vocab = set()

        #go through the training vectors
        for vector in train :  

            #go through each word in each document
            for word in vector[1].split() :

                vocab.add(word) #get vocabulary (set will eliminate repeats)

        #probability that label = pos/neg    
        probpositive = positivecount/(positivecount+negativecount)
        probnegative = negativecount/(positivecount+negativecount)

        #the number of tokens in the training set for positive and negative
        positiveTokens = sum(vocabcountpositive.values())
        negativeTokens = sum(vocabcountnegative.values())
        
        #get size of vocab for negative and positive documents (for smoothing)
        #the number of types in the training set for positive and negative
        positiveTypes = len(vocabcountpositive.values())
        negativeTypes = len(vocabcountnegative.values())

        #count of words for positive and negative (multiply by smoothing in case we change smoothing to non 1 value)
        denompositive = positiveTokens + (positiveTypes * smoothing)
        denomnegative = negativeTokens + (negativeTypes * smoothing)

        #get probability of word being a positive and negative label (count(pos)/count(training) etc)
        #build the probability for each token in our training corpus
        vocabProbPositive = dict()
        vocabProbNegative = dict()

        #for each word in our vocabulary (we can only calculate this value for seen words)
        for word in vocab :

            #if this word has been seen in a positive document
            if word in vocabcountpositive :

                #calculate the P(word | pos) = count(word | pos)/(count(pos tokens) + count(pos types))
                vocabProbPositive[word] = (vocabcountpositive[word]+smoothing) / denompositive
            else :
                #here we have seen the word but not in a positive document
                #so we need to use smoothing to calculate the probability (ow it would be 0)
                #same as above except count(word | pos) = 0 so numerator = smoothing
                vocabProbPositive[word] = smoothing / denompositive

            if word in vocabcountnegative : #same as above except for negative documents
                vocabProbNegative[word] = (vocabcountnegative[word] + smoothing) /denomnegative
            else :
                vocabProbNegative[word] = smoothing / denomnegative

        print("Testing classifier...")

        #counts of correct predictions and incorrect predictions
        tp = 0 #true positive
        fp = 0 #false positive
        tn = 0 #true negative
        fn = 0 #false negative
        
        #go through each test and calculate the actual values, then see which probability is higher
        #then compare to actual and determine prediction accuracy
        for vector in test :

            #we need P(label) * sigma P(words | label) so start with P(label)
            #convert to log space, also need one for both positive and negative
            probdocpositive = math.log(probpositive)
            probdocnegative = math.log(probnegative)
            
            #for each word in the document
            for word in list(set(vector[1].split())) : #converting to set then list removes duplicates and makes our "vector" binary
                
                #if the word actually has a value
                if word in vocabProbPositive :

                    #multiply the probability (addition because of log space)
                    probdocpositive += math.log(vocabProbPositive[word])

                #if the word actually has a value
                if word in vocabProbNegative :

                    #multiply the probability (addition because of log space)
                    probdocnegative += math.log(vocabProbNegative[word])

            #the final probability that a document is a certain label
            #convert back to normal space (out of log space)
            probdocpositive = math.exp(probdocpositive)
            probdocnegative = math.exp(probdocnegative)

            #the predetermined sentiment
            sentiment = int(vector[0])
            
            #if the document is more likely positive (ie we classified it as positive)
            if probdocpositive > probdocnegative :
                
                #if our sentiment == 1 (positive)
                if sentiment :
                    #predicted matches actual (true positive)
                    tp += 1
                else :
                    #predicted does not match actual (false positive)
                    fp += 1
            else :
                if sentiment :
                    #predicted does not match actual (false negative)
                    fn += 1
                else :
                    #predicted matches actual (true negative)
                    tn += 1
            
        print("Test results / metrics:")
        print()
    
        #calculate our test metrics and round to 4 decimal places
        sensitivity = round(tp / (tp + fn), 4)
        specificity = round(tn / (fp + tn), 4)
        precision = round(tp / (tp + fp), 4)
        accuracy = round((tp + tn) / (tp + fp + tn + fn), 4)
        npv = round(tn/(tn + fn), 4)
        f = round(tp / (tp + (.5 * (fp + fn))), 4)

        #display metrics
        print(f"Number of true positives: {tp} \nNumber of true negatives: {tn} \nNumber of false positives: {fp} \nNumber of false negatives: {fn}")
        print(f"Sensitivity (recall): {sensitivity} \nSpecificity: {specificity} \nPrecision: {precision} \nNegative Predictive Value: {npv} \nAccuracy {accuracy} \nF-Score: {f}")

        print()

        #add our general probability of P(doc = label) to our weights (for our analyze inputs file)
        vocabProbPositive["ProbPositive"] = probpositive
        vocabProbNegative["ProbNegative"] = probnegative

        #save our positive and negative weights for analyzeinputs.py to open and use
        with open(posweightfilepath, "w") as weightfile: 
            json.dump(vocabProbPositive, weightfile)
        
        with open(negweightfilepath, "w") as weightfile: 
            json.dump(vocabProbNegative, weightfile)

        #make it so we enter the while loop
        response = "Y"

        #ask for user sentences until they say no
        while response.upper() == "Y" :

            #prompt for sentence
            sentence = input("Enter your sentence: ")

            #analyze the inputs
            output = AnalyzeInputs.analyze(sentence)

            #output of analyzeinputs = [label, probpos, probneg]
            if output[0] == 1 :
                label = "positive"
            else :
                label = "negative"

            #tell the user the classification and probability
            print(f"\nSentence S: \"{sentence}\" was classified as {label}.\nP(positive | S) = {output[1]} \nP(negative | S) = {output[2]}\n")

            #prompt to continue
            response = input("Do you want to enter another sentence [Y/N]?")

main() #only have the one function so called it main