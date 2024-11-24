#%%
import torch
import numpy as np
import random
import matplotlib.pyplot as plt

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)  
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark =  True

def plot_loss_history(train_loss_history, val_loss_history):
    plt.figure(figsize=(10, 5))
    plt.plot(train_loss_history, label='Train Loss')
    plt.plot(val_loss_history, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss History')
    plt.show()

def denormalize(image, mean, std):
    mean = np.array(mean)
    std = np.array(std)
    image = image * std + mean
    return np.clip(image, 0, 1)   

def l1_loss(self, output, target):
    return torch.mean(torch.abs(output - target))

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
    
# reverse_transform = Compose([
#      Lambda(lambda t: (t + 1) / 2),
#      Lambda(lambda t: t.permute(1, 2, 0)), # CHW to HWC
#      Lambda(lambda t: t * 255.),
#      Lambda(lambda t: t.numpy().astype(np.uint8)),
#      ToPILImage(),
# ])