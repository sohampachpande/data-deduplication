import numpy as np
import pandas as pd
import nltk

def editDistance(w1, w2, substitutionCost=2, insertCost=1, deleteCost=1):
    n_=len(w1)+1
    m_=len(w2)+1

    dp = {}
    for i in range(n_): dp[i,0]=i

    for j in range(m_): dp[0,j]=j
    
    for i in range(1, n_):
        for j in range(1, m_):
            cost = 0 if w1[i-1] == w2[j-1] else substitutionCost
            dp[i,j] = min(dp[i, j-1]+insertCost, dp[i-1, j]+deleteCost, dp[i-1, j-1]+cost)

    return dp[i,j]

def isEditDistanceDuplicate(word1, word2, threshold=3):
	editD = editDistance(word1, word2)

	if editD<=threshold and min(len(word1), len(word2))>=threshold:
		return 1.0
	else:
		return 0.0

def cleanText(word):
	return word.lower().strip()

"""
returns if the two words are duplicates using heuristics
"""
def isDuplicateHeuristic(word1, word2, threshold=3):
	w1, w2 = cleanText(word1), cleanText(word2)
	return isEditDistanceDuplicate(w1,w2,threshold)





"""
Applies Heuristic Deduplication on Column of df
returns duplicate word pairs and a new deduplicated column (df)
"""
def heuristicDeduplication_OnColumn(df, colFeature):
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

def heuristicDeduplication_onDF(df, colFeatures):

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
