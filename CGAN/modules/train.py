#%% Import
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import wandb

# Loss function for GAN
class CGANLoss(nn.Module):
    def __init__(self):
        super(CGANLoss, self).__init__()
        self.criterion = nn.BCELoss()

    def forward(self, prediction, target):
        return self.criterion(prediction, target)

#%% CGAN Training Function
def train(
    train_dataloader, 
    model,
    g_optimizer, 
    d_optimizer, 
    config, 
    device
):
    
    num_epoch = config["epochs"]
    noise_size = config["noise_size"]
    batch_size = config["batch_size"]

    gan_loss = CGANLoss()
    model.train()
    
    for epoch in tqdm(range(num_epoch), desc="training..."):
        logs = {
            'd_loss': [],
            'g_loss': [],
            'd_performance': [],
            'g_performance': [],
        }
        
        for i, (inputs, target) in enumerate(train_dataloader):
            real_label = torch.full((batch_size, 1), 1, dtype=torch.float32).to(device)
            fake_label = torch.full((batch_size, 1), 0, dtype=torch.float32).to(device)

            inputs = inputs.reshape(batch_size, -1).to(device)
            target = target.to(device)
            
            # +---------------------+
            # |   Train Generator   |
            # +---------------------+
            """ train w. real"""
            g_optimizer.zero_grad()

            noise = torch.randn([batch_size, noise_size], device=device)
            gen_image = model.generator(noise, target)
            g_loss = gan_loss(model.discriminator(gen_image, target), real_label)
            g_loss.backward()
            g_optimizer.step()

            # +---------------------+
            # | Train Discriminator |
            # +---------------------+
            """ train w. real """
            d_optimizer.zero_grad()
            
            z = torch.randn([batch_size, noise_size], device=device)
            fake_image = model.generator(z, target)
            fake_loss = gan_loss(model.discriminator(fake_image, target), fake_label)
            real_loss = gan_loss(model.discriminator(inputs, target), real_label)
            d_loss = (fake_loss + real_loss) / 2

            d_loss.backward()
            d_optimizer.step()

            d_performance = model.discriminator(fake_image, target).mean().item()
            g_performance = model.discriminator(inputs, target).mean().item()

            logs['d_loss'].append(d_loss.item())
            logs['g_loss'].append(g_loss.item())
            logs['d_performance'].append(d_performance)
            logs['g_performance'].append(g_performance)

        if (epoch + 1) % 50 == 0:
            print(f"Epoch [{epoch + 1}/{num_epoch}]")
            print(f"Discriminator Loss: {np.mean(logs['d_loss'])}, Generator Loss: {np.mean(logs['g_loss'])}")
            print(f"Discriminator performance: {np.mean(logs['d_performance']):.2f}, Generator performance: {np.mean(logs['g_performance']):.2f}")

        wandb.log({
            'epoch': epoch + 1,
            'd_loss': np.mean(logs['d_loss']),
            'g_loss': np.mean(logs['g_loss']),
            'd_performance': np.mean(logs['d_performance']),
            'g_performance': np.mean(logs['g_performance'])
        })
        
    return