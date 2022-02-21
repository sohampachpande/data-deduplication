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

from .utils import isDuplicateHeuristic

import pandas as pd
import torch.nn as nn
import torch.nn.functional as F

class RuleBasedModel():
    def __init__(self, data_path, threshold=3):
        df = pd.read_csv(
            data_path, usecols=['w1', 'w2', 'isDuplicate']
            ).sample(frac=1).reset_index(drop=True)

        df['w1'] = df['w1'].astype(str)
        df['w2'] = df['w2'].astype(str)
        
        self.df_train = df[:int(0.7*len(df))]
        self.df_test = df[int(0.7*len(df)):]

        self.threshold = threshold

    def test(self):
        y_pred = self.df_test.apply(
            lambda x: isDuplicateHeuristic(
                x['w1'], x['w2'], threshold=self.threshold)
                ,axis=1)

        return y_pred

class NN(nn.Module):
    def __init__(self, d):
        super().__init__()
        
        self.fc1 = nn.Linear(d*2, 100)
        self.fc2 = nn.Linear(100, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.sigmoid(self.fc3(x))
        return x