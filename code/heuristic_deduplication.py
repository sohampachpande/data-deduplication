import numpy as np
import pandas as pd
import nltk


def isEditDistanceDuplicate(word1, word2, threshold=3):
  editD = nltk.edit_distance(word1, word2)

  if editD<threshold:
    return True
  return False


def heuristic_deduplication(df, colFeatures):

  for colFeature in colFeatures:

    df['Original {}'.format(colFeature)] = df[colFeature]

    # Lower Case and Strip
    df[colFeature]=df.apply(lambda x: x[colFeature].lower().strip() if type(x)==str else x, axis=1)


    OriginalFeatureCount = df[colFeature].value_counts()
    setFeatures = set(list(df[colFeature]))
    edit_dist_pairs = []


    for i,r1 in df.iterrows():
      for w in setFeatures:
        # print(r1[colFeature], w, df[colFeature].value_counts()[r1[colFeature]], df[colFeature].value_counts()[w])
        if type(r1[colFeature])!=str and type(w)!=str:
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