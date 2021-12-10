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