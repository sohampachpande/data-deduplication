import numpy as np
import pandas as pd

def split_train_test_data(data,split=0.2):
  train_df, test_df = train_test_split(data, test_size=split)
  return (train_df,test_df)
  
def remove_nan_dominant_cols(train_df):
  drop_cols=[]
  for i in (train_df.columns):
    if(train_df[i].isna().sum()>200000):
      drop_cols.append(i)
  return train_df.drop(drop_cols, 1)