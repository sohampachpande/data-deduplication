import pandas as pd
import numpy as np
from tqdm import tqdm_notebook as tqdm
import itertools
from nltk.util import ngrams 
from sklearn.feature_extraction.text import CountVectorizer
from strsimpy import Jaccard,Cosine,WeightedLevenshtein

def generate_n_grams(word,n):
    return list(ngrams(word,n))

def generateNGramDF(path):
    df=pd.read_csv(path)

    df['w1'] = df['w1'].astype(str)
    df['w2'] = df['w2'].astype(str)

    df['w1']=df['w2'].apply(lambda x:x.lower())
    df['w2']=df['w2'].apply(lambda x:x.lower())
    df['w1_number_of_words']=df['w1'].apply(lambda x:len(list(x.split(' '))))
    df['w2_number_of_words']=df['w2'].apply(lambda x:len(list(x.split(' '))))


    alphabets='abcdefghijklmnopqrstuvwxyz '
    bigram_list=list(itertools.permutations(list(alphabets),2))
    def unigram_count(word, ch):
        count=0
        for c in word:
            if(c==ch):
                count+=1
        return count

    for w in ['w1', 'w2']:
        for ch in list(alphabets):
            df[w+'_unigram_{}'.format(ch)] = df[w].apply(lambda x: x.count(ch))

    for w in ['w1', 'w2']:
        for ch in list(bigram_list):
            df[w+'_bigram_{}'.format(ch[0]+ch[1])] = df[w].apply(lambda x: x.count(ch[0]+ch[1]))

    return df


def generateDistanceMetricData(path):
    df=pd.read_csv(path)

    df['w1'] = df['w1'].astype(str)
    df['w2'] = df['w2'].astype(str)

    df['w1']=df['w1'].apply(lambda x:x.lower())
    df['w2']=df['w2'].apply(lambda x:x.lower())

    j=Jaccard(1)
    df['jaccard_1'] = df.apply(lambda x: j.distance(x.w1, x.w2), axis=1)
    j=Jaccard(2)
    df['jaccard_2'] = df.apply(lambda x: j.distance(x.w1, x.w2), axis=1)
    j=Jaccard(3)
    df['jaccard_3'] = df.apply(lambda x: j.distance(x.w1, x.w2), axis=1)


    cosine=Cosine(1)
    df['cosine_1'] = df.apply(lambda x: cosine.distance(x.w1, x.w2), axis=1)
    cosine=Cosine(2)
    df['cosine_2'] = df.apply(lambda x: cosine.distance(x.w1, x.w2), axis=1)
    cosine=Cosine(3)
    df['cosine_3'] = df.apply(lambda x: cosine.distance(x.w1, x.w2), axis=1)

    def get_substitution_cost():
        return 2
    def get_insertion_cost():
        return 1
    def get_deletion_cost():
        return 1
    lev=WeightedLevenshtein()
    df['w_levenshtein'] = df.apply(lambda x: lev.distance(x.w1, x.w2), axis=1)

    return df