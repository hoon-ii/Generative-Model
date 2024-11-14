#%%
import torch
import random
import numpy as np

"""for reproducibility"""
def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 모든 GPU에 대한 시드 고정
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # NumPy 시드 고정
    np.random.seed(seed)
    random.seed(seed) 

def make_beta_schedule(schedule='linear',timesteps=1_000,beta_start=0.0001,beta_end=0.02):
    if schedule == 'linear':
        betas = torch.linspace(beta_start, beta_end, timesteps)
    elif schedule == 'quad':
        betas = torch.linspace(beta_start ** 0.5, beta_end ** 0.5, timesteps) ** 2
    elif schedule == 'sigmoid':
        x = torch.linspace(-6, 6, timesteps)
        betas = torch.sigmoid(x) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule}")    
    return betas  


def normal_kl(mean1, logvar1, mean2, logvar2):
    """
    KL divergence between two normal distributions parameterized by mean and log-variance.
    
    Args:
    - mean1, logvar1: Mean and log-variance of the first distribution.
    - mean2, logvar2: Mean and log-variance of the second distribution.
    
    Returns:
    - KL divergence between the two distributions.
    """
    # Calculate the KL divergence
    kl_div = 0.5 * (-1.0 + logvar2 - logvar1 + torch.exp(logvar1 - logvar2)
                    + (mean1 - mean2).pow(2) * torch.exp(-logvar2))
    
    return kl_div.mean()