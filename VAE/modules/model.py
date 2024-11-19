#%%
import torch
from torch import nn
from torch.utils.data import DataLoader
import math
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
#%% VAE(fc lyaer로 구성)
class VAE(nn.Module):
    def __init__(self, config, EncodedInfo):
        super(VAE, self).__init__()
        self.config = config
        self.channels = EncodedInfo.channels
        self.height = EncodedInfo.height
        self.width = EncodedInfo.width
        self.input_dim = self.channels*self.height*self.width

        # Encoder
        en = []
        in_dim = self.input_dim

        for h in config["hidden_dims"]:
            en.append(nn.Linear(in_dim, h))
            en.append(nn.ReLU())
            in_dim = h
        en.append(nn.Linear(h, config["latent_dim"] * 2)) # mu, logvar 공간을 마련하기 위해 *2 해줌

        # 신경망을 self.encoder 속성으로 정의
        self.encoder = nn.Sequential(*en)
        
        # Decoder
        de = []
        in_dim = config["latent_dim"]
        for h in reversed(config["hidden_dims"]):
            de.append(nn.Linear(in_dim, h))
            de.append(nn.ReLU())
            in_dim = h
        de.append(nn.Linear(h, self.input_dim))
        de.append(nn.Sigmoid())

        # 신경망을 self.decoder 속성으로 정의
        self.decoder = nn.Sequential(*de)

    def encode(self, x):
        encoded = x.view(-1, self.input_dim) 
        mu_logvar = self.encoder(encoded)
        return mu_logvar
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
    
    def decode(self, z):
        decoded = self.decoder(z)
        decoded = decoded.view(-1, self.channels, self.height, self.width)
        return decoded
    
    def forward(self, x):
        mu_logvar = self.encode(x)
        mu, logvar = mu_logvar.chunk(2, dim = 1)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar
    
    # def generate(self, test_dataset, device):
    #     test_dataloader = DataLoader(
    #         test_dataset, 
    #         batch_size=64,
    #     )
    #     batch, label = next(iter(test_dataloader))
        
    #     with torch.no_grad():
    #         batch, label = batch.to(device), label.to(device)
    #         recon, mu, logvar = self(batch)
       
    #     grid = gridspec.GridSpec(3, 3)
    #     plt.figure(figsize = (10, 10))
    #     plt.subplots_adjust(wspace = 0.5, hspace = 0.5)
    #     for i in range(9):    
    #         ax = plt.subplot(grid[i])
    #         plt.imshow(recon[i].reshape(self.height, self.width).cpu().detach().numpy(), cmap='gray')
    #         ax.get_xaxis().set_visible(False)
    #         ax.get_yaxis().set_visible(False)
    #         ax.set_title('label : {}'.format(label[i]))

    #     return ax

    def generate(self, test_dataset, num_samples, device):
        test_dataloader = DataLoader(
            test_dataset, 
            batch_size=num_samples,
        )
        batch, _ = next(iter(test_dataloader))
        
        with torch.no_grad():
            batch = batch.to(device)
            generated_images, _, _ = self(batch)
        return generated_images