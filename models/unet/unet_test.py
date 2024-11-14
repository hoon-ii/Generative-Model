import torch
import torch.nn as nn
import math

def get_timestep_embedding(timesteps, embedding_dim):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models:
    From Fairseq.
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
    assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0,1,0,0))
    return emb

class Normalize(nn.Module):
    def __init__(self, in_channels, num_groups=32):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)

    def forward(self, x):
        return self.norm(x)

class ResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None, temb_channels=512, dropout=0.0):
        super().__init__()
        out_channels = in_channels if out_channels is None else out_channels
        self.norm1 = Normalize(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        
        if temb_channels > 0:
            self.temb_proj = nn.Linear(temb_channels, out_channels)
        
        self.norm2 = Normalize(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x, temb):
        h = self.norm1(x)
        h = torch.nn.functional.silu(h)
        h = self.conv1(h)
        
        if temb is not None:
            h = h + self.temb_proj(torch.nn.functional.silu(temb))[:, :, None, None]
        
        h = self.norm2(h)
        h = torch.nn.functional.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)
        
        return h + self.shortcut(x)

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, ch, ch_mult=(1, 2, 4, 8), num_res_blocks=2, 
                 attn_resolutions=(16,), dropout=0.0, resamp_with_conv=True, resolution=64):
        super().__init__()
        self.ch = ch
        self.temb_ch = ch * 4
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution

        # timestep embedding
        self.temb = nn.Module()
        self.temb.dense = nn.ModuleList([
            nn.Linear(ch, self.temb_ch),
            nn.Linear(self.temb_ch, self.temb_ch),
        ])

        # Downsampling
        self.conv_in = nn.Conv2d(in_channels, ch, kernel_size=3, stride=1, padding=1)
        
        curr_res = resolution
        in_ch_mult = (1,) + tuple(ch_mult)
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(block_in, block_out, temb_channels=self.temb_ch, dropout=dropout))
                block_in = block_out
            self.down.append(block)
            if i_level != self.num_resolutions - 1:
                self.down.append(nn.AvgPool2d(kernel_size=2))

        # Middle
        self.mid = nn.ModuleList([
            ResnetBlock(block_in, block_out, temb_channels=self.temb_ch, dropout=dropout),
            ResnetBlock(block_out, block_out, temb_channels=self.temb_ch, dropout=dropout)
        ])

        # Upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            block_in = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                block.append(ResnetBlock(block_in, block_out, temb_channels=self.temb_ch, dropout=dropout))
                block_in = block_out
            self.up.append(block)
            if i_level != 0:
                self.up.append(nn.Upsample(scale_factor=2, mode='nearest'))

        # End
        self.norm_out = Normalize(block_in)
        self.conv_out = nn.Conv2d(block_in, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x, t=None):
        # Timestep embedding
        temb = get_timestep_embedding(t, self.ch)
        temb = self.temb.dense[0](temb)
        temb = torch.nn.functional.silu(temb)
        temb = self.temb.dense[1](temb)

        # Downsampling
        hs = [self.conv_in(x)]
        for block in self.down:
            if isinstance(block, nn.AvgPool2d):
                hs.append(block(hs[-1]))
            else:
                h = block[0](hs[-1], temb)
                hs.append(h)
        
        # Middle
        h = hs[-1]
        for block in self.mid:
            h = block(h, temb)

        # Upsampling
        for block in self.up:
            if isinstance(block, nn.Upsample):
                h = block(h)
            else:
                h = block[0](torch.cat([h, hs.pop()], dim=1), temb)

        # End
        h = self.norm_out(h)
        h = torch.nn.functional.silu(h)
        return self.conv_out(h)
