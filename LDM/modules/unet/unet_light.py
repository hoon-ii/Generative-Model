import torch
import torch.nn as nn
import torch.nn.functional as F
from models.modules.attention import SelfAttention, CrossAttention

class TimeEmbedding(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.linear_1 = nn.Linear(n_embd, n_embd)
        self.linear_2 = nn.Linear(n_embd, n_embd)

    def forward(self, t):
        t = t.float()
        t = self.linear_1(t)
        t = F.silu(t)
        t_emb = self.linear_2(t)
        return t_emb

class UNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_groups=1, use_attention=False, num_heads=2):
        super(UNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.group_norm1 = nn.GroupNorm(num_groups, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.group_norm2 = nn.GroupNorm(num_groups, out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.use_attention = use_attention
        if use_attention:
            self.attention = SelfAttention(out_channels, num_heads)

    def forward(self, x, timestep_emb=None):
        if timestep_emb is not None:
            timestep_emb = timestep_emb.view(x.size(0), -1, 1, 1)
            timestep_emb = timestep_emb.expand_as(x)
            x = x + timestep_emb
        x = self.relu(self.group_norm1(self.conv1(x)))
        x = self.relu(self.group_norm2(self.conv2(x)))
        if self.use_attention:
            x = self.attention(x)
        return x

class LightUNet(nn.Module):
    def __init__(self, in_channels, out_channels, context_dim=256, num_groups=1, num_heads=2, timestep_dim=32):
        super(LightUNet, self).__init__()

        self.time_embedding = TimeEmbedding(timestep_dim)

        self.encoder1 = UNetBlock(in_channels, 32, num_groups, use_attention=True, num_heads=num_heads)
        self.encoder1_cross_attention = CrossAttention(32, context_dim, num_heads)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder2 = UNetBlock(32, 64, num_groups, use_attention=True, num_heads=num_heads)
        self.encoder2_cross_attention = CrossAttention(64, context_dim, num_heads)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck_block = UNetBlock(64, 128, num_groups, use_attention=True, num_heads=num_heads)
        self.bottleneck_attention = SelfAttention(128, num_heads)
        self.bottleneck_cross_attention = CrossAttention(128, context_dim, num_heads)

        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder2_block = UNetBlock(128, 64, num_groups, use_attention=True, num_heads=num_heads)
        self.decoder2_cross_attention = CrossAttention(64, context_dim, num_heads)

        self.upconv1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.decoder1_block = UNetBlock(64, 32, num_groups, use_attention=True, num_heads=num_heads)
        self.decoder1_cross_attention = CrossAttention(32, context_dim, num_heads)

        self.final_conv = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, x, context, t):
        timestep_emb = self.time_embedding(t)

        enc1 = self.encoder1(x, timestep_emb)
        enc1 = self.encoder1_cross_attention(enc1, context)
        enc1_pooled = self.pool1(enc1)

        enc2 = self.encoder2(enc1_pooled, timestep_emb)
        enc2 = self.encoder2_cross_attention(enc2, context)
        enc2_pooled = self.pool2(enc2)

        bottleneck = self.bottleneck_block(enc2_pooled, timestep_emb)
        bottleneck = self.bottleneck_attention(bottleneck)
        bottleneck = self.bottleneck_cross_attention(bottleneck, context)

        dec2 = self.upconv2(bottleneck)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2_block(dec2, timestep_emb)
        dec2 = self.decoder2_cross_attention(dec2, context)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1_block(dec1, timestep_emb)
        dec1 = self.decoder1_cross_attention(dec1, context)

        final_output = self.final_conv(dec1)

        return final_output
