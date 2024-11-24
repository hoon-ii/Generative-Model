#%%
#%% Import necessary libraries
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils
import numpy as np
import matplotlib.pyplot as plt
import wandb
import os

from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR 

loss_fn = nn.MSELoss()

def train_vqvae(config, ddpm, optimizer, train_loader):
    logs = {
            'vqvae_loss': [],
        }
    device = config['device']
    ddpm.vqvae.train()
    ddpm.unet.eval()
    for epoch in tqdm(range(1, config['epochs'] + 1), desc="VQVAE is under training"):
        train_loss = 0.0
        ema_loss = None 
        for imgs, texts in train_loader:
            origin_img = imgs.to(device)
            embed_loss, recon_img, perplexity = ddpm.vqvae(origin_img)
            loss = loss_fn(origin_img, recon_img) + embed_loss 

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if ema_loss is None:
                ema_loss = loss.item()  
            else:
                ema_loss = 0.999 * ema_loss + 0.001  * loss.item()  

            logs['vqvae_loss'].append(loss.item())

        wandb.log({
            'epoch': epoch + 1,
            'vqvae_loss': np.mean(logs['vqvae_loss']),
        }) 

        if (epoch + 1) % 50 == 0:
            print(f"Epoch [{epoch + 1}/{config['epochs']}]")
            print(f"VQVAE Loss: {np.mean(logs['vqvae_loss'])}")
                  
def train_unet(config, ddpm, optimizer, train_loader, text_embedder):
    logs = {
            'unet_loss': [],
        }
    device = config['device']
    ddpm.vqvae.eval()
    ddpm.unet.train()
    for epoch in tqdm(range(1, config['epochs'] + 1), desc='UNET is under training..'):
        ema_loss = None
        for imgs, texts in train_loader:
            origin_img = imgs.to(device)
            
            emb_loss, z_0, _, _, _ = ddpm.vqvae.encode(origin_img)
            context = text_embedder.generate_text_embeddings(texts).to(device)
            t = torch.randint(0, config["timesteps"], (z_0.size(0),), device=device, dtype=torch.long)
            noisy_z, noise = ddpm.q_sample(z_0, t)  
            
            optimizer.zero_grad()
            
            pred_e_t = ddpm.unet(noisy_z, context, t)
            
            pred_z_0 = ddpm.predict_start_from_noise(noisy_z, t, pred_e_t)
            
            recon_img = ddpm.vqvae.decode(pred_z_0)
            
            loss_recon = loss_fn(origin_img, recon_img) + emb_loss

            loss_noise = loss_fn(pred_e_t, noise)  

            loss = loss_recon + loss_noise
            loss.backward()
            
            optimizer.step()

            logs['unet_loss'].append(loss.item())

        wandb.log({
            'epoch': epoch,
            'unet_loss': np.mean(logs['unet_loss']),
        })

        if (epoch + 1) % 50 == 0:
            print(f"Epoch [{epoch}/{config['epochs']}], UNET Loss: {np.mean(logs['unet_loss'])}")

                  
#%% Load Loader
