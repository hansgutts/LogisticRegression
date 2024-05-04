<h1>Binary Bag of Words Sentiment Analysis Through Logistic Regression From Scratch</h1>

This project is still a work in progress. Feature extraction and hyperparamater tuning still needs to be done.

<h2>Purpose</h2>
The initial purpose of this was to see the performance of logistic regression in sentiment analysis (in terms of prediction 
accuracy) when compared to a similar but simpler method such as a naive bayes classifier (look here https://github.com/hansgutts/NaiveBayes 
for naive bayes implementation and performance). As this project evolved, I ran into other issues that needed to be adressed and 
became more about solving the problems of large datasets and (eventually) feature extraction.

<h2>Initial Logistic Regression Implementation</h2>
My initial approach for implementing logistic regression by hand used only built in Python data types. The class "logistic" within
LogisticRegression.py has 6 class variables and 4 functions that actually perform logistic regression. The 6 fields are 
<ul>
  <li>Learning Rate - the rate at which the weights adjust (default = .01) </li>
  <li>Iterations - the number of times it runs the same training data and adjusts weights (default = 1000)</li>
  <li>Weights - the values that each feature is weighted</li>
  <li>Bias - value added after weight * X</li>
  <li>Losses - list to keep track of the loss between iterations (not currently used)</li>
  <li>Threshold - the value the sigmoid function must surpass to be considered a positive element</li>
</ul>

The 4 functions are

<h3>sigmoid(numpy.array x) -> numpy.array</h3>
The sigmoid function takes in an array of values and returns a list of each inputs sigmoid function value (1/(1+e^-x)).
This gets us a value between 0-1 and acts as a probability value.

<h3>computeloss(numpy.array pred, numpy.array actual) -> float</h3>
The compute loss function calculates the error based on the predicted values and actual values and returns a value for the average loss. 

<h3>feedforward(numpy.array X) -> numpy.array</h3>
Feed forward takes in a list of vectors X and returns the raw sigmoid value of each vectors dot product with weights.

<h3>fit(numpy.array values, numpy.array actual) -> none</h3>
Fit takes in a list of feature vectors (values) and a corresponding list of classifications (actual). This returns none as weights and 
bias are updated within the class.

<h3>predict(numpy.array X) -> numpy.array</h3>
Predict takes in a list of feature vectors and returns a list of predicted classifications (1 or 0). 

<h2>Dataset</h2>
The dataset I used is the UCI_Drug (https://www.kaggle.com/datasets/arpikr/uci-drug/data) dataset found on Kaggle. In short, 
it has reviews of drugs that treat varying medicial conditions. The dataset includes uniqueID, drugName, condition, review, 
rating, date, and usefulCount but the only dimensions I used for sentiment analysis were review and rating. Rating was converted 
to a negative sentiment (0) if the rating was < 7 or a positive sentiment (1) if the rating was >= 7. This cut off point was 
initially arbitrarially decided. The dataset initially came in two tsv files (UCIdrug_test.csv and UCIdrug_train.csv) though 
these had to be combined for different training/testing splits. There are 215063 samples after preprocessing with 142306 rated 
as positive and 72757 rated as negative, which does create an unbalanced dataset in favor of positive ratings. 

<h2>Preprocessing</h2>
Preprocessing
As discussed, the initial datasets needed to be condensed into a single csv file. The actual review entries contained html tags 
in the form "&----;" that needed to be transformed to real characters in order to have correct words. Punctuation was then removed 
in its entirety as we are only focused on the words in the review. I then also removed the name of the drug being reviewed from 
the actual user review as I was worried the classifier would be overtuned toward negative or positive classifications based on 
whether the drug itself was generally rated positive or negative (basically, I thought the weight on drug names would be larger 
than other words. This did not end up being the case, but I left the drug names out). Preprocessing was completed in the file named 
"CleanCSV.py". We did not do any additional preprocessing such as lemmatization, stemming, or removal of stop words. 

