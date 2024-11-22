#%%
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
import os
from torchvision.utils import save_image, make_grid
import wandb
#%%
# def train_function(
#         model, 
#         config, 
#         optimizer,
#         train_dataloader, 
#         device):
def train_function(
        model, 
        config, 
        optimizer,
        train_dataloader, 
        test_dataloader,
        device):

    save_dir = './img_CVAE'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
                    
    for epoch in range(config['epochs']):
        logs = {
            'mse_loss': [],
            # 'bce_loss' : [],
            'kl_loss': [],
            'loss': [],
        }

        for batch, labels in tqdm(train_dataloader, desc="inner loop..."):
            loss_ = []
            batch, labels = batch.to(device), labels.to(device)
            recon_batch, mu, logvar = model(batch, labels)
            mse_loss = F.mse_loss(recon_batch, batch)
            # bce_loss = F.binary_cross_entropy(recon, batch)
            kl_loss = torch.mean(
                -0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim = 1), dim = 0
            )
            loss = mse_loss + kl_loss
            # loss = bce_loss + kl_loss
            
            loss_.append(('mse_loss', mse_loss))
            # loss_.append(('bce_loss', bce_loss))
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
        
        if (epoch + 1) % 5 == 0:
            model.eval()
            with torch.no_grad():
                for batch_te, labels_te in test_dataloader:
                    batch_te, labels_te = batch_te.to(device), labels_te.to(device)
                    generated_images, _, _ = model(batch_te, labels_te)
                grid = make_grid(generated_images.cpu(), nrow=10, normalize=True)

                # 이미지 저장 경로
                save_path = os.path.join(save_dir, f'CVAE_epoch_{epoch + 1:03d}.png')
                save_image(grid, save_path)
                print(f"Generated images saved at: {save_path}")

        """update log"""
        wandb.log({x : np.mean(y) for x, y in logs.items()})

    return
# %%
