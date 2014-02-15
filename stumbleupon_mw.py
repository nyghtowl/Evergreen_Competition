from sklearn import metrics,preprocessing,cross_validation
from sklearn.feature_extraction.text import TfidfVectorizer
import sklearn.linear_model as lm
import numpy as np
import pandas as p

def load_data(filename, col):
    print "loading data.."
    return np.array(p.read_table(filename))[:,col]

def vectorize(vectorizer):
    '''
    Converts values between 0-1 - normalizes
    '''
    return verctorizer(min_df=3,  max_features=None, strip_accents='unicode',  
        analyzer='word',token_pattern=r'\w{1,}',ngram_range=(1, 2), use_idf=1,smooth_idf=1,sublinear_tf=1)

def main():
    traindata = load_data('./data/train.tsv', 2) 
    testdata = load_data('./data/test.tsv', 2)
    y = load_data('./data/train.tsv', -1)
    tfv = vectorize(TfidVectorizer)