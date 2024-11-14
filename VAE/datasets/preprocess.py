#%%
from torch.utils.data import Dataset
from datasets.raw_data import load_MNIST


from collections import namedtuple
EncodedInfo = namedtuple(
    'EncodedInfo', 
    ['channels', 'height', 'width'])

#%%

#%%
class CustomMNIST(Dataset):
    def __init__(self, train=True):
        train_dataset, test_dataset = load_MNIST()
        self.data = train_dataset if train else test_dataset
        
        self.c, self.h, self.w = self.data[0][0].shape
        self.EncodedInfo = EncodedInfo(
            self.c, self.h, self.w)
            
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):        
        return self.data[idx]
#%%