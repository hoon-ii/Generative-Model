#%%
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import wandb

# Loss function for GAN
class GANLoss(nn.Module):
    def __init__(self):
        super(GANLoss, self).__init__()
        self.criterion = nn.BCELoss()

    def forward(self, prediction, target):
        return self.criterion(prediction, target)

#%%
def train(
    train_dataloader, 
    model,
    g_optimizer, 
    d_optimizer, 
    config, 
    device
):
    """Training loop for GAN"""
    num_epoch = config["epochs"]
    noise_size = config["noise_size"]
    batch_size = config["batch_size"]

    gan_loss = GANLoss()
    model.train()
    for epoch in range(num_epoch):
        logs = {
            'd_loss': [],
            'g_loss': [],
            'd_performance': [],
            'g_performance': [],
        }
        
        for i, (images, _) in tqdm(enumerate(train_dataloader), desc="(GAN) training..."):
            real_label = torch.full((batch_size, 1), 1, dtype=torch.float32).to(device)
            fake_label = torch.full((batch_size, 1), 0, dtype=torch.float32).to(device)

            real_images = images.view(batch_size, -1).to(device)

            # +---------------------+
            # |   Train Generator   |
            # +---------------------+
            g_optimizer.zero_grad()
            
            z = torch.randn(batch_size, noise_size).to(device)
            fake_images = model.gen_step(z)

            g_loss = gan_loss(model.disc_step(fake_images), real_label)

            g_loss.backward()
            g_optimizer.step()

            # +---------------------+
            # | Train Discriminator |
            # +---------------------+
            d_optimizer.zero_grad()
            
            fake_loss = gan_loss(model.disc_step(fake_images.detach()), fake_label)
            real_loss = gan_loss(model.disc_step(real_images), real_label)
            d_loss = (fake_loss + real_loss) / 2

            d_loss.backward()
            d_optimizer.step()

            d_performance = model.disc_step(real_images).mean().item()
            g_performance = model.disc_step(fake_images).mean().item()

            logs['d_loss'].append(d_loss.item())
            logs['g_loss'].append(g_loss.item())
            logs['d_performance'].append(d_performance)
            logs['g_performance'].append(g_performance)

            if (i + 1) % 150 == 0:
                print("Epoch [{}/{}], Step [{}/{}], d_loss: {:.5f}, g_loss: {:.5f}"
                      .format(epoch + 1, num_epoch, i + 1, len(train_dataloader), d_loss.item(), g_loss.item()))

        print("Epoch [{}] Discriminator performance: {:.2f}, Generator performance: {:.2f}"
              .format(epoch + 1, np.mean(logs['d_performance']), np.mean(logs['g_performance'])))

        wandb.log({
            'epoch': epoch + 1,
            'd_loss': np.mean(logs['d_loss']),
            'g_loss': np.mean(logs['g_loss']),
            'd_performance': np.mean(logs['d_performance']),
            'g_performance': np.mean(logs['g_performance'])
        })
        
    return
