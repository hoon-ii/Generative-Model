#%%
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
#%%
class VAE(nn.Module):
    def __init__(self, config, EncodedInfo, device):
        super(VAE, self).__init__()
        
        self.config = config
        self.EncodedInfo = EncodedInfo
        self.device = device
        
        self.cont_dim = self.EncodedInfo.num_continuous_features
        self.disc_dim = sum(self.EncodedInfo.num_categories)
        self.p = self.cont_dim + self.disc_dim
        self.hidden_dim = 128
        
        """encoder"""
        self.encoder = nn.Sequential(
            nn.Linear(self.p, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, config["latent_dim"] * 2),
        ).to(device)
        
        """decoder"""
        self.delta = torch.arange(0, 1 + config["step"], step=config["step"]).view(1, -1).to(device)
        self.M = self.delta.size(1) - 1
        self.decoder = nn.Sequential(
            nn.Linear(config["latent_dim"] + self.p, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.cont_dim * (1 + (self.M + 1)) + self.disc_dim),
        ).to(device)
        
    def get_posterior(self, input):
        h = self.encoder(input)
        mean, logvar = torch.split(h, self.config["latent_dim"], dim=1)
        return mean, logvar
    
    def sampling(self, mean, logvar):
        noise = torch.randn(mean.size(0), self.config["latent_dim"]).to(self.device) 
        z = mean + torch.exp(logvar / 2) * noise
        return z
    
    def encode(self, input):
        mean, logvar = self.get_posterior(input)
        z = self.sampling(mean, logvar)
        return z, mean, logvar
    
    def quantile_parameter(self, z, condition):
        h = self.decoder(torch.cat([z, condition], dim=-1))
        logit = h[:, -self.disc_dim:] # categorical
        spline = h[:, :-self.disc_dim] # continuous
        h = torch.split(spline, 1 + (self.M + 1), dim=1)
        gamma = [h_[:, [0]] for h_ in h]
        beta = [torch.cat([
            torch.zeros_like(gamma[0]),
            nn.Softplus()(h_[:, 1:]) # positive constraint
        ], dim=1) for h_ in h]
        beta = [b[:, 1:] - b[:, :-1] for b in beta]
        return gamma, beta, logit
    
    def quantile_function(self, alpha, gamma, beta, j):
        return gamma[j] + (beta[j] * torch.where(
            alpha - self.delta > 0,
            alpha - self.delta,
            torch.zeros(()).to(self.device)
        )).sum(dim=1, keepdims=True)
        
    def quantile_inverse(self, x, gamma, beta):
        C = self.EncodedInfo.num_continuous_features
        delta_ = self.delta.unsqueeze(2).repeat(1, 1, self.M + 1) # [1, M+1, M+1]
        delta_ = torch.where(
            delta_ - self.delta > 0,
            delta_ - self.delta,
            torch.zeros(()).to(self.device)) # [1, M+1, M+1]
        
        beta_delta = (torch.stack(beta, dim=2) * delta_.unsqueeze(2).unsqueeze(4)).sum(dim=3).squeeze(0)
        mask = torch.cat(gamma, dim=1).unsqueeze(1) + beta_delta.permute([1, 0, 2])
        mask = torch.where(
            mask <= x[:, :C].unsqueeze(1), 
            mask, 
            torch.zeros(()).to(self.device)).type(torch.bool).type(torch.float)
        alpha_tilde = x[:, :C] - torch.cat(gamma, dim=1)
        alpha_tilde += (mask * torch.stack(beta, dim=2) * self.delta.unsqueeze(2)).sum(dim=1)
        alpha_tilde /= (mask * torch.stack(beta, dim=2)).sum(dim=1) + self.config["threshold"] # numerical stability
        alpha_tilde = torch.clip(alpha_tilde, 0, 1) # numerical stability
        return alpha_tilde
    
    def forward(self, input, condition):
        z, mean, logvar = self.encode(input)
        gamma, beta, logit = self.quantile_parameter(z, condition)
        return z, mean, logvar, gamma, beta, logit
    
    @torch.no_grad()
    def generate_data(self, n, diffusion, train_dataset):
        batch_size = 64
        data = []
        steps = n // batch_size + 1
        
        for _ in tqdm(range(steps), desc="Generate Synthetic Dataset..."):
            input_batch = torch.zeros(
                batch_size, self.p
            ).to(self.device)
            batch = torch.zeros(
                batch_size, train_dataset.EncodedInfo.num_features
            ).to(self.device)
            randn = torch.from_numpy(
                diffusion.sample(batch_size, self.config["latent_dim"]).numpy()
            ).to(self.device).to(torch.float)
            
            cont_dim = self.EncodedInfo.num_continuous_features
            # permute the generation order of columns
            # the classification target is always the last one
            for j in torch.cat([
                torch.randperm(train_dataset.EncodedInfo.num_features - 1),
                torch.tensor([train_dataset.EncodedInfo.num_features - 1]) # downstream task target column
            ]):
                gamma, beta, logit = self.quantile_parameter(randn, input_batch)
                
                if j < cont_dim:
                    alpha = torch.rand(batch_size, 1).to(self.device)
                    Q = self.quantile_function(alpha, gamma, beta, j) ### inverse transform sampling
                    input_batch[:, [j]] = Q
                    batch[:, [j]] = Q
                else:
                    st = 0
                    if j > cont_dim:
                        st = st + sum(self.EncodedInfo.num_categories[:j - cont_dim])
                    dim = self.EncodedInfo.num_categories[j - cont_dim]
                    ed = st + dim
                    out = Categorical(logits=logit[:, st : ed]).sample()
                    input_batch[:, cont_dim + st : cont_dim + ed] = F.one_hot(out.long(), num_classes=dim)
                    batch[:, j] = out
            data.append(batch)
                
        data = torch.cat(data, dim=0).to(float)
        data = data[:n, :]
        data = pd.DataFrame(
            data.cpu().numpy(), 
            columns=train_dataset.features)
        
        """un-standardization of synthetic data"""
        for col, scaler in train_dataset.scalers.items():
            data[[col]] = scaler.inverse_transform(data[[col]])
        
        """post-process"""
        data[train_dataset.categorical_features] = data[train_dataset.categorical_features].astype(int)
        data[train_dataset.integer_features] = data[train_dataset.integer_features].round(0).astype(int)
        
        return data
#%%