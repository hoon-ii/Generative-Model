#%%
import numpy as np
from tqdm import tqdm
import wandb

import torch

from modules.train1 import onehot
#%%
def anneal_lr(step, optimizer, config):
    frac_done = step / config["n_iters"]
    lr = config["lr2"] * (1 - frac_done)
    for param_group in optimizer.param_groups:
        param_group["lr2"] = lr

def train_function(model, diffusion, train_dataloader, config, optimizer, device):
    
    for epoch in tqdm(range(config["n_iters"]), desc="training..."):
        logs = {
            'diffusion_loss': [], 
        }
        
        for x_batch in iter(train_dataloader):
            x_batch = x_batch.to(device)
            
            """Latent variables from aggregated posterior distribution"""
            with torch.no_grad():
                input_batch = onehot(x_batch, model)
                _, x, _ = model.encode(input_batch) # mean vector
            
            loss_ = []
            
            optimizer.zero_grad()
            
            loss = diffusion.loss(x)
            
            loss_.append(('diffusion_loss', loss))
            
            loss.backward()
            optimizer.step()
            
            anneal_lr(epoch, optimizer, config)
                
            """accumulate losses"""
            for x, y in loss_:
                logs[x] = logs.get(x) + [y.item()]
        
        if epoch % 10 == 0:
            print_input = f"Epoch [{epoch+1:03d}/{config['n_iters']}]"
            print_input += "".join(
                [", {}: {:.4f}".format(x, np.mean(y)) for x, y in logs.items()]
            )
            print(print_input)
        
        """update log"""
        wandb.log({x : np.mean(y) for x, y in logs.items()})
        
    return
#%%