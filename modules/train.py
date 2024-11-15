#%%
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F

import wandb
#%%
def train_function(
        model, 
        config, 
        optimizer,
        train_dataloader, 
        device):

    for epoch in range(config['epochs']):
        logs = {
            'bce_loss': [],
            'kl_loss': [],
            'loss': [],
        }

        for batch, _ in tqdm(train_dataloader, desc="inner loop..."):
            loss_ = []
            
            batch = batch.to(device)

            recon, mu, logvar = model(batch)
            bce_loss = F.binary_cross_entropy(recon, batch, reduction = 'sum')
            kl_loss = torch.mean(
                -0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim = 1), dim = 0
            )
            loss = bce_loss + kl_loss
        
            loss_.append(('bce_loss', bce_loss))
            loss_.append(('kl_loss', kl_loss))
            loss_.append(('loss', loss))
            
            optimizer.zero_grad()
            loss.backward()

            optimizer.step()
            
            """accumulate losses"""
            for x, y in loss_:
                logs[x] = logs.get(x) + [y.item()]
            

        print_input = "[epoch {:03d}]".format(epoch + 1)
        print_input += ''.join([', {}: {:.4f}'.format(x, np.mean(y)) for x, y in logs.items()])
        print(print_input)
        
        """update log"""
        wandb.log({x : np.mean(y) for x, y in logs.items()})
    return
# %%
