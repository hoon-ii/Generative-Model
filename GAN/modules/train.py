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
    generator, 
    discriminator, 
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
    for epoch in range(num_epoch):
        logs = {
            'd_loss': [],
            'g_loss': [],
            'd_performance': [],
            'g_performance': [],
        }
        
        for i, (images, _) in tqdm(enumerate(train_dataloader), desc="(GAN) training..."):
            # Set labels for real and fake data
            real_label = torch.full((batch_size, 1), 1, dtype=torch.float32).to(device)
            fake_label = torch.full((batch_size, 1), 0, dtype=torch.float32).to(device)

            # Prepare real images
            real_images = images.view(batch_size, -1).to(device)

            # +---------------------+
            # |   Train Generator   |
            # +---------------------+
            g_optimizer.zero_grad()
            
            # Generate fake images and calculate generator loss
            z = torch.randn(batch_size, noise_size).to(device)
            fake_images = generator(z)

            g_loss = gan_loss(discriminator(fake_images), real_label)

            # Backpropagation for generator
            g_loss.backward()
            g_optimizer.step()

            # +---------------------+
            # | Train Discriminator |
            # +---------------------+
            d_optimizer.zero_grad()
            
            # Calculate discriminator loss for fake and real images
            fake_loss = gan_loss(discriminator(fake_images.detach()), fake_label)
            real_loss = gan_loss(discriminator(real_images), real_label)
            d_loss = (fake_loss + real_loss) / 2

            # Backpropagation for discriminator
            d_loss.backward()
            d_optimizer.step()

            # Track discriminator and generator performance
            d_performance = discriminator(real_images).mean().item()
            g_performance = discriminator(fake_images).mean().item()

            # Append losses to logs
            logs['d_loss'].append(d_loss.item())
            logs['g_loss'].append(g_loss.item())
            logs['d_performance'].append(d_performance)
            logs['g_performance'].append(g_performance)

            if (i + 1) % 150 == 0:
                print("Epoch [{}/{}], Step [{}/{}], d_loss: {:.5f}, g_loss: {:.5f}"
                      .format(epoch + 1, num_epoch, i + 1, len(train_dataloader), d_loss.item(), g_loss.item()))

        # Print performance metrics at the end of each epoch
        print("Epoch [{}] Discriminator performance: {:.2f}, Generator performance: {:.2f}"
              .format(epoch + 1, np.mean(logs['d_performance']), np.mean(logs['g_performance'])))

        # Log to wandb at the end of each epoch
        wandb.log({
            'epoch': epoch + 1,
            'd_loss': np.mean(logs['d_loss']),
            'g_loss': np.mean(logs['g_loss']),
            'd_performance': np.mean(logs['d_performance']),
            'g_performance': np.mean(logs['g_performance'])
        })
        
    return