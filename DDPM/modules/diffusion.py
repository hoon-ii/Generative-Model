#%%
import math
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
#%%
"""
Based in part on: https://github.com/lucidrains/denoising-diffusion-pytorch/blob/5989f4c77eafcdc6be0fb4739f0f277a6dd7f7d8/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py#L281
"""
eps = 1e-8

def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    """
    Get a pre-defined beta schedule for the given name.
    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """
    if schedule_name == "linear":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.00001
        beta_end = scale * 0.01
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif schedule_name == "cosine":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].
    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)
#%%
class Diffusion(nn.Module):
    def __init__(
        self,
        denoise_fn,
        num_timesteps=1000,
        scheduler="linear",
        device=torch.device('cpu'),
    ):
        super(Diffusion, self).__init__()
        self.denoise_fn = denoise_fn
        self.num_timesteps = num_timesteps
        self.device = device
        
        self.betas = torch.tensor(
            get_named_beta_schedule(scheduler, num_timesteps)
        ).to(device).float()
        self.alphas = 1. - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        self.alpha_bars_prev = F.pad(self.alpha_bars[:-1], (1, 0), value = 1.)
        
        # reverse process
        self.posterior_variance = self.betas * (1. - self.alpha_bars_prev) / (1. - self.alpha_bars)
        self.posterior_log_variance_clipped = (self.posterior_variance.clamp(min=1e-20)).log()
        self.posterior_mean_coef1 = self.betas * self.alpha_bars_prev.sqrt() / (1. - self.alpha_bars)
        self.posterior_mean_coef2 = (1. - self.alpha_bars_prev) * self.alphas.sqrt() / (1. - self.alpha_bars)
        
    def extract(self, a, t, x_shape):
        b, *_ = t.shape
        t = t.to(a.device)
        out = a.gather(-1, t)
        while len(out.shape) < len(x_shape):
            out = out[..., None]
        return out.expand(x_shape)
        
    def loss(self, x): # x = x_0
        """1. sampling timestep"""
        t = torch.randint(0, self.num_timesteps, (x.size(0), ), device=self.device).long()
        
        """2. forward diffusion: q(x_t|x_0)"""
        noise = torch.randn_like(x) # epsilon
        x_t = self.extract(self.alpha_bars.sqrt(), t, x.shape) * x
        x_t += self.extract((1. - self.alpha_bars).sqrt(), t, x.shape) * noise
        
        """3. noise prediction"""
        noise_pred = self.denoise_fn(x_t, t)
        
        """4. variational bound"""
        loss = nn.MSELoss()(noise, noise_pred)
        
        return loss
    
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
    def sample(self, n, d_in):
        x_t = torch.randn((n, d_in), device=self.device) # start with pure noise (x_T)
        for i in reversed(range(0, self.num_timesteps)):
            t = torch.full((n, ), i, device=self.device).long()
            noise_pred = self.denoise_fn(x_t, t)
            x_0 = self.predict_start_from_noise(x_t, t, noise=noise_pred) # estimated x_0
            x_t = self.gaussian_p_sample(x_start=x_0, x_t=x_t, t=t)
        return x_t
    
    @torch.no_grad()
    # Algorithm 2 in 
    # Ho, J., Jain, A., & Abbeel, P. (2020). 
    # Denoising diffusion probabilistic models. 
    # Advances in neural information processing systems, 33, 6840-6851.
    def sample_(self, n, d_in):
        x_t = torch.randn((n, d_in), device=self.device) # start with pure noise (x_T)
        for i in reversed(range(0, self.num_timesteps)):
            t = torch.full((n, ), i, device=self.device).long()
            noise_pred = self.denoise_fn(x_t, t)
            sample = x_t - self.extract((1. - self.alphas) / (1. - self.alpha_bars).sqrt(), t, x_t.shape) * noise_pred
            sample /= self.extract(1 / self.alphas.sqrt(), t, x_t.shape)
            if i > 0: # if t = 0, there is no noise
                noise = torch.randn_like(x_t).to(self.device)
                # sample += self.extract(self.betas.sqrt(), t, x_t.shape) * noise # sigma_t^2 = beta_t
                sample += self.extract(self.posterior_variance.sqrt(), t, x_t.shape) * noise
            x_t = sample
        return x_t
#%%