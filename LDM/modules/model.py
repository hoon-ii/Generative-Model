#%%
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from tqdm import tqdm

from modules.vqvae.vqvae import VQVAE 
from modules.unet.unet import UNet 

def make_beta_schedule(schedule='linear',timesteps=1_000,beta_start=0.0001,beta_end=0.02):
    if schedule == 'linear':
        betas = torch.linspace(beta_start, beta_end, timesteps)
    elif schedule == 'quad':
        betas = torch.linspace(beta_start ** 0.5, beta_end ** 0.5, timesteps) ** 2
    elif schedule == 'sigmoid':
        x = torch.linspace(-6, 6, timesteps)
        betas = torch.sigmoid(x) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule}")    
    return betas  

class DDPM(nn.Module):
    def __init__(self, config):
        super(DDPM, self).__init__()
        self.config = config

        self.unet = UNet(
         in_channels=config['embedding_dim'],
         base_channels=config['base_channels'],
         ch_mult=config['ch_mult'],
         attention_res=config['attention_res'],
         context_dim=config['n_embeddings'],
         num_heads=config['num_heads'],
         num_groups=config['num_groups'],
         cur_res=config['cur_res']
        ).to(config["device"])
        self.vqvae = VQVAE(config)
        
        self.num_timesteps = config["timesteps"]
        self.device = config["device"]
        
        self.betas = torch.tensor(
            make_beta_schedule(config["scheduler"], self.num_timesteps)
        ).to(self.device).float()
        self.alphas = 1. - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        self.alpha_bars_prev = F.pad(self.alpha_bars[:-1], (1, 0), value = 1.)
        
        # reverse process
        self.posterior_variance = self.betas * (1. - self.alpha_bars_prev) / (1. - self.alpha_bars)
        self.posterior_log_variance_clipped = (self.posterior_variance.clamp(min=1e-20)).log()
        self.posterior_mean_coef1 = self.betas * self.alpha_bars_prev.sqrt() / (1. - self.alpha_bars)
        self.posterior_mean_coef2 = (1. - self.alpha_bars_prev) * self.alphas.sqrt() / (1. - self.alpha_bars)

        self.sqrt_alphas_cumprod = torch.sqrt(self.alpha_bars)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alpha_bars).to(config["device"])

    def loss(self, x, context): # x = x_0
        """1. sampling timestep"""
        t = torch.randint(0, self.num_timesteps, (x.size(0), ), device=self.device).long()
        
        """2. forward diffusion: q(x_t|x_0)"""
        noise = torch.randn_like(x) # epsilon
        x_t = self.extract(self.alpha_bars.sqrt(), t, x.shape) * x
        x_t += self.extract((1. - self.alpha_bars).sqrt(), t, x.shape) * noise
        
        """3. noise prediction"""
        noise_pred = self.unet(x=x_t,context=context, t=t)
        
        """4. variational bound"""
        loss = nn.MSELoss()(noise, noise_pred) #L2
        
        """5. kl loss? """
        # x_0_recon = self.predict_start_from_noise(x_t, t, noise=noise_pred)
        # x_0 =  self.predict_start_from_noise(x_t, t, noise=noise)
        # mean_q, logvar_q = self.gaussian_p_mean_variance(x_0, x_t, t )
        # mean_p, logvar_p = self.gaussian_p_mean_variance(x_0_recon, x_t, t)
        # # p = p.clamp(min=1e-8)
        # # q = q.clamp(min=1e-8)
        # loss_kl = normal_kl(mean_p, logvar_p, mean_q, logvar_q)

        return loss #+ loss_kl
    
    def q_sample(self, x_start, t, noise=None):
        """
        forward diffusion process

        Args:
            x_start: start value for the forward process
            t: current timestep
            noise: noise to add to the start value
        Returns:
            x_start with added noise corresponding to the current timestep
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = self.extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self.extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise, noise
    
    def extract(self, a, t, x_shape):
        """
        extracts an appropriate t index for a batch of indices
        """
        b = t.shape[0]
        out = a.gather(-1, t.to(a.device))
        return out.reshape(b, *((1,) * (len(x_shape) - 1))).to(t.device)

    def gaussian_p_mean_variance(self, x_start, x_t, t):
        model_mean = self.extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
        model_mean += self.extract(self.posterior_mean_coef2, t, x_t.shape) * x_t 
        model_log_variance = self.extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return model_mean, model_log_variance
    
    def gaussian_p_sample(self, x_start, x_t, t):
        model_mean, model_log_variance = self.gaussian_p_mean_variance(x_start, x_t, t)
        noise = torch.randn_like(x_t).to(self.device)
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x_t.shape) - 1)))
        ) # if t = 0, there is no noise
        sample = model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise
        return sample
    
    def predict_start_from_noise(self, x_t, t, noise):
        return (
            self.extract(1. / self.alpha_bars.sqrt(), t, x_t.shape) * x_t -
            self.extract((1. / self.alpha_bars - 1.).sqrt(), t, x_t.shape) * noise
        )
    
    @torch.no_grad()
    def sample(self, n, d_in, context):
        n , C, H, W = d_in
        x_t = torch.randn((n, C, H, W), device=self.device) # start with pure noise (x_T)
        for i in tqdm(reversed(range(0, self.num_timesteps)),desc='under sampling..'):
            t = torch.full((n, ), i, device=self.device).long()
            noise_pred = self.denoise_fn(x_t, context, t)
            x_0 = self.predict_start_from_noise(x_t, t, noise=noise_pred) # estimated x_0
            x_t = self.gaussian_p_sample(x_start=x_0, x_t=x_t, t=t)
        return x_t

