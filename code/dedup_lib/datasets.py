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

from torch.utils.data import Dataset
import torch

class dedupDataset(Dataset):
    """User defined class to build a dataset using Pytorch class Dataset."""
    
    def __init__(self, data, encoding, transform = None):
        """Method to initilaize variables.""" 
        self.data = data
        self.encoding = encoding

        self.transform=transform

    def __getitem__(self, index):
        w1, w2, isDuplicate = self.data.loc[index]
        
        e1 = self.encoding[w1.lower().strip()]
        e2 = self.encoding[w2.lower().strip()]

        E = torch.cat((torch.tensor(e1),torch.tensor(e2)))        

        if self.transform is not None:
            E = self.transform(E)

        return E, torch.tensor(isDuplicate).float()

    def __len__(self):
        return len(self.data)