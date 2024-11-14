#%%
import numpy as np
from tqdm import tqdm
import wandb

import torch
from torch import nn
import torch.nn.functional as F
#%%
def CRPS_loss(model, x_batch, alpha_tilde, gamma, beta):
    C = model.EncodedInfo.num_continuous_features
    delta = model.delta.unsqueeze(-1) # [1, M+1, 1]
    
    term = (1 - delta.pow(3)) / 3 - delta - torch.maximum(alpha_tilde.unsqueeze(1), delta).pow(2) # [batch, M+1, p]
    term += 2 * torch.maximum(alpha_tilde.unsqueeze(1), delta) * delta # [batch, M+1, p]
    
    crps = (2 * alpha_tilde) * x_batch[:, :C] # [batch, p]
    crps += (1 - 2 * alpha_tilde) * torch.cat(gamma, dim=1) # [batch, p]
    crps += (torch.stack(beta, dim=2) * term).sum(dim=1) # [batch, p]
    crps *= 0.5
    return crps

def onehot(batch, model):
    cont_dim = model.EncodedInfo.num_continuous_features
    onehot_batch = []

    # continuous
    onehot_batch.extend(
        [batch[:, :cont_dim]]
    )
    # categorical
    batch_ = batch.clone()
    for j, dim in enumerate(model.EncodedInfo.num_categories):
        onehot_batch.append(
            F.one_hot(
                batch_[:, cont_dim + j].long(),
                num_classes=dim
            ) 
        )
    onehot_batch = torch.cat(onehot_batch, dim=1)
    return onehot_batch

def masking(batch, mask, model):
    cont_dim = model.EncodedInfo.num_continuous_features
    masked_batch = []

    # continuous
    batch_ = batch.clone()
    batch_[~mask] = 0.
    masked_batch.append(
        batch_[:, :cont_dim]
    )
    # categorical
    batch_ = batch.clone()
    batch_[~mask] = torch.nan
    for j, dim in enumerate(model.EncodedInfo.num_categories):
        masked_batch.append(
            F.one_hot(
                batch_[:, cont_dim + j].nan_to_num(dim).long(),
                num_classes=dim+1
            )[:, :-1] # remove NaN index column
        )
    masked_batch = torch.cat(masked_batch, dim=1)
    return masked_batch
#%%
def train_function(model, train_dataloader, config, optimizer, device):
    
    for epoch in tqdm(range(config["epochs"]), desc="training..."):
        logs = {
            'loss': [], 
            'recon': [],
            'KL': [],
            'activated': []
        }
        
        for x_batch in iter(train_dataloader):
            x_batch = x_batch.to(device)
            
            mask = torch.rand_like(x_batch) > torch.rand(x_batch.size(0), 1).to(device)
            # downstream task target column is the last column 
            # target colunm is always masked
            mask[:, -1] = False 
            
            input_batch = onehot(x_batch, model)
            conditional_batch = masking(x_batch, mask, model)
            
            optimizer.zero_grad()
            
            z, mean, logvar, gamma, beta, logit = model(input_batch, conditional_batch)
            
            loss_ = []
            
            """1. Reconstruction loss"""
            cont_dim = model.EncodedInfo.num_continuous_features
            ### continuous
            alpha_tilde = model.quantile_inverse(input_batch, gamma, beta)
            recon = CRPS_loss(model, input_batch, alpha_tilde, gamma, beta)
            recon = recon[~mask[:, :cont_dim]]
            recon = recon.sum() / (~mask[:, :cont_dim]).sum()
            ### categorical
            st = 0
            for j, dim in enumerate(model.EncodedInfo.num_categories):
                ed = st + dim
                targets = x_batch[:, cont_dim + j].long()
                out = logit[:, st : ed]
                CE = nn.CrossEntropyLoss()(
                    out[~mask[:, j]], 
                    targets[~mask[:, j]]
                )
                if not CE.isnan():
                    recon += CE
                st = ed
            loss_.append(('recon', recon))
            
            """2. KL-Divergence"""
            KL = torch.pow(mean, 2).sum(dim=1)
            KL -= logvar.sum(dim=1)
            KL += torch.exp(logvar).sum(dim=1)
            KL -= config["latent_dim"]
            KL *= 0.5
            KL = KL.mean()
            loss_.append(('KL', KL))
            
            """3. ELBO"""
            loss = recon + config["beta"] * KL 
            loss_.append(('loss', loss))
            
            var_ = logvar.exp() < 0.1
            loss_.append(('activated', var_.float().mean()))
            
            loss.backward()
            optimizer.step()
                
            """accumulate losses"""
            for x, y in loss_:
                logs[x] = logs.get(x) + [y.item()]
        
        if epoch % 10 == 0:
            print_input = f"Epoch [{epoch+1:03d}/{config['epochs']}]"
            print_input += "".join(
                [", {}: {:.4f}".format(x, np.mean(y)) for x, y in logs.items()]
            )
            print(print_input)
        
        """update log"""
        wandb.log({x : np.mean(y) for x, y in logs.items()})
        
    return
#%%