#%%
import torch
import numpy as np
import random

def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 모든 GPU에 대한 시드 고정
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # NumPy 시드 고정
    np.random.seed(seed)
    random.seed(seed) 

def denormalize(tensor):
    """
    Tensor 데이터를 Normalize 이전 상태로 복원.
    Args:
        tensor: torch.Tensor, Normalize된 데이터.
        mean: float, Normalize 과정에서 사용한 평균값.
        std: float, Normalize 과정에서 사용한 표준편차.
    """
    tensor = tensor.clone()
    tensor.mul_(0.3081).add_(0.1307)  # 반대로 Normalize 복원
    return tensor