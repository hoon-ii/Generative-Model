#%%
import torch
import torch.nn as nn
import numpy as np

import wandb
from tqdm import tqdm

class Unet_loss(nn.Module):
    def __init__(self, config):
        super(Unet_loss, self).__init__()
        self.mse_loss = nn.MSELoss(reduction="mean")
        self.device = config["device"]

    def forward(self, x_recon, x):
        recon_loss = self.mse_loss(x_recon, x)
        return recon_loss

#%%
def train_unet(
    train_dataloader, 
    unet,
    optimizer, 
    config, 
    device):
    
    """iterative optimization"""
    loss_function = Unet_loss(config)
    for epoch in range(config["epochs"]):
        logs = {
            'loss' : [],
            'embedding_loss': [],
            'recon_loss' : [],
        }
    
        for i, batch in tqdm(enumerate(train_dataloader), desc="(UNet)training..."):
            loss_ = []
            batch = batch.to(device)
            optimizer.zero_grad()
            
            # Forward pass for UNet
            embedding_loss, x_hat, perplexity = unet(batch)
            recon_loss = loss_function(x_hat, batch)
            loss = embedding_loss + recon_loss
            
            loss.backward(retain_graph=True)
            optimizer.step()
            loss_.append(('loss', loss))
            loss_.append(('embedding_loss', embedding_loss))
            loss_.append(('recon_loss', recon_loss))   

            """accumulate losses"""
            for x, y in loss_:
                logs[x] = logs.get(x) + [y.item()] 
        
        print_input = "[epoch {:03d}]".format(epoch + 1)
        print_input += ''.join([', {}: {:.4f}'.format(x, np.sum(y)) for x, y in logs.items()])
        print(print_input)

        """update log"""
        wandb.log({x : np.sum(y) for x, y in logs.items()}) 
    return
