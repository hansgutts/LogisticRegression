import pandas as pd
import pickle

corpusPath = 'UCIDrugClean.csv'
exportPath = 'UCIDrugVocab.pickle'

def CreateVocab(export = True) :

    #vocabulary is set to immediately eliminate duplicate words in vocab
    vocab = set()

    dataset = pd.read_csv(corpusPath, names=['Review', 'Sentiment'], header=None)

    for values in dataset.itertuples() :
        for word in values.Review.split() :
            vocab.add(word)

    print(len(vocab))

    if export :
        with open(exportPath, 'wb') as pickleFile : #pickle logic/info from https://stackoverflow.com/questions/11218477/how-can-i-use-pickle-to-save-a-dict-or-any-other-python-object
            pickle.dump(vocab, pickleFile, pickle.HIGHEST_PROTOCOL)
    
    else :
        return vocab
    
CreateVocab(True)