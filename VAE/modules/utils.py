
#%%
import torch
import numpy as np
import random

def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    np.random.seed(seed)
    random.seed(seed) 

def denormalize(tensor):
    """ only for MNIST dataset """
    tensor = tensor.clone()
    tensor.mul_(0.3081).add_(0.1307)   
    return tensor