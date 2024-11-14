#%%
import torch
from torch.utils.data import Dataset

from sklearn.model_selection import train_test_split
from datasets.raw_data import load_raw_data

#%%
class CustomDataset(Dataset):
    def __init__(self, config, train=True):
        self.config = config
        self.train = train

        data, continuous_features, categorical_features, integer_features, ClfTarget = load_raw_data(self.config["dataset"])
        self.continuous_features = continuous_features
        self.categorical_features = categorical_features
        self.integer_features = integer_features
        self.ClfTarget = ClfTarget
        
        self.features = self.continuous_features + self.categorical_features
        
        # encoding for categorical variables.
        data[self.categorical_features] = data[self.categorical_features].apply(
            lambda col: col.astype('category').cat.codes)
        
        data = data[self.features]
        train_data, test_data = train_test_split(
            data, test_size=config["test_size"], random_state=config["seed"]
        )

        data = train_data if self.train else test_data
        self.raw_data = train_data[self.features] if train else test_data[self.features]
        self.data = data.reset_index(drop=True)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.data[idx])
    
