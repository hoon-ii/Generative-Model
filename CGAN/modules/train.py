#%% Import
import torch
import torch.nn as nn
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
    """
    Training loop for Conditional GAN (CGAN).
    
    Args:
        train_dataloader: DataLoader for training data.
        model: CGAN model (with generator and discriminator).
        g_optimizer: Optimizer for the generator.
        d_optimizer: Optimizer for the discriminator.
        config: Dictionary of configuration parameters.
        device: Device to run the training (e.g., 'cuda' or 'cpu').
    """
    num_epoch = config["epochs"]
    noise_size = config["noise_size"]
    batch_size = config["batch_size"]

    gan_loss = CGANLoss()
    model.train()
    
    for epoch in range(num_epoch):
        logs = {
            'd_loss': [],
            'g_loss': [],
            'd_performance': [],
            'g_performance': [],
        }
        
        for i, (images, labels) in tqdm(enumerate(train_dataloader), desc=f"Epoch {epoch + 1}/{num_epoch}"):
            # Prepare real and fake labels
            real_label = torch.full((batch_size, 1), 1, dtype=torch.float32).to(device)
            fake_label = torch.full((batch_size, 1), 0, dtype=torch.float32).to(device)

            # Prepare real images and labels
            real_images = images.to(device)
            real_labels = labels.to(device)

            # +---------------------+
            # |   Train Generator   |
            # +---------------------+
            g_optimizer.zero_grad()
            d_optimizer.zero_grad()
            
            # Generate fake images with labels
            z = torch.randn(batch_size, noise_size).to(device)
            fake_labels = torch.randint(0, config["num_classes"], (batch_size,), device=device)
            fake_images = model.generator(z, fake_labels)

            # Generator loss (fooling the discriminator)
            g_loss = gan_loss(model.discriminator(fake_images, fake_labels), real_label)
            g_loss.backward()
            g_optimizer.step()

            # +---------------------+
            # | Train Discriminator |
            # +---------------------+
            g_optimizer.zero_grad()
            d_optimizer.zero_grad()
            
            # Discriminator loss for real and fake images
            real_validity = model.discriminator(real_images, real_labels)
            fake_validity = model.discriminator(fake_images.detach(), fake_labels)
            
            real_loss = gan_loss(real_validity, real_label)
            fake_loss = gan_loss(fake_validity, fake_label)
            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            d_optimizer.step()

            # Log discriminator and generator performance
            d_performance = real_validity.mean().item()
            g_performance = fake_validity.mean().item()

            logs['d_loss'].append(d_loss.item())
            logs['g_loss'].append(g_loss.item())
            logs['d_performance'].append(d_performance)
            logs['g_performance'].append(g_performance)

        # Print progress every 10 epochs
        if (epoch + 1) % 100 == 0:
            print(f"Epoch [{epoch + 1}/{num_epoch}]")
            print(f"Discriminator Loss: {np.mean(logs['d_loss'])}, Generator Loss: {np.mean(logs['g_loss'])}")
            print(f"Discriminator performance: {np.mean(logs['d_performance']):.2f}, Generator performance: {np.mean(logs['g_performance']):.2f}")

        # Log metrics to WandB
        wandb.log({
            'epoch': epoch + 1,
            'd_loss': np.mean(logs['d_loss']),
            'g_loss': np.mean(logs['g_loss']),
            'd_performance': np.mean(logs['d_performance']),
            'g_performance': np.mean(logs['g_performance'])
        })
        
    return
