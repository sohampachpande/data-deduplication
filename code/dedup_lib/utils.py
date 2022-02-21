#Copyright 2020 Soham Pachpande, Gehan Chopade, Arun Kumar
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.

import numpy as np
import pandas as pd
import nltk
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score, precision_score, recall_score

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
input : 
y_pred = Predicted Labels
y_truth = Ground Truth Labels
output : 
Confusion Matrix with  accuracy, recall, F1-score and precision values 
Function computes and visualises Confusion Matrix along with 
accuracy, recall, F1-score and precision values
"""
def makeCFwithStats(y_pred, y_truth):
    cf = confusion_matrix(y_truth, y_pred)
    # accuracy  = accuracy_score(y_truth, y_pred)
    # precision = precision_score(y_truth, y_pred)
    # recall    = recall_score(y_truth, y_pred)
    # f1_score_  = f1_score(y_truth, y_pred)
    # stats_text = "\n\nAccuracy={:0.3f}\nPrecision={:0.3f}\nRecall={:0.3f}\nF1 Score={:0.3f}".format(accuracy,precision,recall,f1_score_)

    stats_text = ''

    df_cm = pd.DataFrame(cf, index = ["Not Duplicate", "Duplicate"],
                    columns = ["Not Duplicate", "Duplicate"])
    plt.figure(figsize = (6,5))
    sn.heatmap(df_cm/np.sum(cf), annot=True, fmt='.2%')
    plt.ylabel('True label', fontsize=14)
    plt.xlabel('Predicted label' + stats_text, fontsize=14)
    plt.show()
