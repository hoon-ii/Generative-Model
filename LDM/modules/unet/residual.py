#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.modules.attention import SinusoidalPositionEmbeddings

class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,
                 num_groups=4, dropout, temb_channels=512):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups, out_channels),
            nn.SiLU()
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups, out_channels),
            nn.SiLU()
        )

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x, t):
        identity = self.shortcut(x)

        x = self.block1(x)
        B, C, H, W = x.shape
        # condition with time if required
        if t is not None:
            time_emb = SinusoidalPositionEmbeddings(H*W)(t)
            time_emb = time_emb.unsqueeze(1)  # Reshape for broadcasting to (b, dim, 1)
            time_emb = time_emb.repeat(1,C,1)
            time_emb = time_emb.view(x.shape)  # Reshape back to (b, c, h, w)
            x = x + time_emb  # Add time embedding to the input feature map
            x = nn.SiLU()(x)

        x = self.block2(x)

        return x + identity


# I made it. However, not used... :( 
class ResdualStack(nn.Module):
    def __init__(self, num_res_blocks, in_channels, out_channels, temb_channels=512, dropout=0.0):
        super(ResdualStack, self).__init__()
        self.num_res_blocks = num_res_blocks
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.temb_channels = temb_channels
        self.dropout = dropout
        
        self.blocks = nn.ModuleList()
        
        block_in = self.in_channels
        for i_block in range(self.num_res_blocks):
            block_out = self.out_channels if isinstance(self.out_channels, int) else self.out_channels[i_block]
            self.blocks.append(ResnetBlock(in_channels=block_in,
                                             out_channels=block_out,
                                             temb_channels=self.temb_channels,
                                             dropout=self.dropout))
            block_in = block_out

    def forward(self, x, temb):
        for block in self.blocks:
            x = block(x, temb)
        return x