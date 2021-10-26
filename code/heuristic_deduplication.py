import numpy as np
import pandas as pd
import nltk


def isEditDistanceDuplicate(word1, word2, threshold=3):
  editD = nltk.edit_distance(word1, word2)

  if editD<threshold:
    return True
  return False



def heuristic_deduplication_col(df, colFeature):
  d={}
  OriginalFeatureCount = df[colFeature].value_counts()
  d[colFeature] = {}
  originalFeatures = np.asarray(list(df[colFeature]))
  NewFeatures = np.asarray(list(map(lambda x:x.lower().strip() if type(x)==str else x, originalFeatures)))

  word_pairs = []
  word_pair_dict = {}
  for w1 in set(NewFeatures):
    for w2 in set(NewFeatures):
      if type(w1)==np.str_ and type(w2)==np.str_ and w1!=w2:
        if w2 in w1:
          word_pairs.append((w1,w2))
          word_pair_dict[w1]=w2
        elif isEditDistanceDuplicate(w1, w2) and OriginalFeatureCount[w1] <  OriginalFeatureCount[w2]:
          word_pairs.append((w1,w2))
          word_pair_dict[w1]=w2
        else:
          continue

  for w1,w2 in word_pair_dict.items():
    NewFeatures[np.where(NewFeatures==w1)]=w2

  return word_pair_dict, NewFeatures

def heuristic_deduplication_ondf(df, colFeatures):

  for colFeature in colFeatures:

    OriginalFeatureCount = df[colFeature].value_counts()

    df['Original {}'.format(colFeature)] = df[colFeature]

    # Lower Case and Strip
    df[colFeature]=df.apply(lambda x: x[colFeature].lower().strip() if type(x)==str else x, axis=1)

    setFeatures = set(list(df[colFeature]))
    edit_dist_pairs = []


    for i,r1 in df.iterrows():
      for w in setFeatures:
        # print(r1[colFeature], w, df[colFeature].value_counts()[r1[colFeature]], df[colFeature].value_counts()[w])
        if type(r1[colFeature])!=str or type(w)!=str:
          continue
        # Check for substring
        if r1[colFeature]!=w and w in r1[colFeature]:
          df.at[i,colFeature]=w
          continue       
        
        # Check for edit distance threshold
        if isEditDistanceDuplicate(r1[colFeature], w):
          if  OriginalFeatureCount[r1[colFeature]] <  OriginalFeatureCount[w]:
            df.at[i,colFeature]=w

  return df
