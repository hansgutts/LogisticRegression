import pandas as pd
import string
import html


filepath = "UCIDrug.csv"
uniquePunctuation = ['\'', '\'', ':', '-', ';']
cleanFilePath = "UCIDrugClean.csv"

replacementDict = {}
drugnameList = []

print("\nWorking...\n")

#read in the dataset as a pandas df
dataset = pd.read_csv(filepath, header=0)

#get a list of all drug names to remove later
for drug in dataset['drugName'].unique() :
    for drugname in drug.replace(' ', '').split('\\') :
        drugnameList.append(drugname)

for punct in string.punctuation :
    replacementDict[punct] = ' '

for punct in uniquePunctuation :
    replacementDict[punct] = ''

print(replacementDict)

#remove unnecessary data columns and null values
dataset = dataset.drop(columns=['uniqueID', 'drugName', 'condition', 'date',
                                 'usefulCount']).dropna() #rating and review remain


#want to transform ratings to 1 for positive 0 for negative
dataset['rating'] = dataset['rating'].transform(lambda x: 1 if x > 7 else 0)

#want to modfy our reviews to translate html tags to punct, lowercase, and remove punctuation.
dataset['review'] = dataset['review'].astype(str).transform(lambda x: html.unescape(x).lower().translate(str.maketrans(replacementDict)))


print(dataset)

dataset.to_csv(cleanFilePath)



