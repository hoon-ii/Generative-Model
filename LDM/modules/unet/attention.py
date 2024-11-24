#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math
#%%
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

def Normalize(in_channels, num_groups=1):
    return torch.nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, 
                              eps=1e-6, affine=True)

class SelfAttention(nn.Module):
    def __init__(self, in_channels, use_time_emb=False):
        super().__init__()
        self.in_channels = in_channels
        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)
        self.use_time_emb = use_time_emb
    def forward(self, x, t=None):
        h_ = x
        h_ = self.norm(h_)
        
        b,c,h,w = h_.shape
        
        if self.use_time_emb is not False:
            time_emb = SinusoidalPositionEmbeddings(h*w)(t)
            time_emb = time_emb.unsqueeze(1)  # Reshape for broadcasting to (b, dim, 1)
            time_emb = time_emb.repeat(1,h_.size(1),1)
            time_emb = time_emb.view(h_.size())  # Reshape back to (b, c, h, w)
            h_ = h_ + time_emb  # Add time embedding to the input feature map
            h_ = nn.SiLU()(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)    

        # compute attention
        q = rearrange(q, 'b c h w -> b (h w) c')
        k = rearrange(k, 'b c h w -> b c (h w)')
        w_ = torch.einsum('bij,bjk->bik', q, k)

        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = rearrange(v, 'b c h w -> b c (h w)')
        w_ = rearrange(w_, 'b i j -> b j i')
        h_ = torch.einsum('bij,bjk->bik', v, w_)
        h_ = rearrange(h_, 'b c (h w) -> b c h w', h=h)
        h_ = self.proj_out(h_)
        return x+h_

# << Run with Positional Embedding(Sinusoidal)
class CrossAttention(nn.Module):
    def __init__(self, embed_dim, context_dim, num_heads=4):
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim  
        self.context_dim = context_dim  
        self.inner_dim = embed_dim * num_heads

        self.scale = (embed_dim // num_heads) ** -0.5
        
        # Linear projections
        self.query_proj = nn.Linear(embed_dim, self.inner_dim, bias=False)
        self.key_proj = nn.Linear(context_dim, self.inner_dim, bias=False)
        self.value_proj = nn.Linear(context_dim, self.inner_dim, bias=False)
        self.out_proj = nn.Linear(self.inner_dim, embed_dim)

    def forward(self, x, context, t):
        B, C, H, W = x.shape
        
        N = context.shape[1]  # Sequence length
        
        t = torch.randint(1, 2, (B,), device='cuda', dtype=torch.long) #Fixed t

        time_emb = SinusoidalPositionEmbeddings(H*W)(time=t) # random_time(positional emb)
        time_emb = time_emb.unsqueeze(1)  # Reshape for broadcasting to (b, dim, 1)
        time_emb = time_emb.repeat(1,C,1)
        time_emb = time_emb.view(x.size())  # Reshape back to (b, c, h, w)
        x = x + time_emb  # Add time embedding to the input feature map
        x = nn.SiLU()(x)
        # Reshape and project queries, keys, values
        queries = self.query_proj(x.view(B, C, -1).permute(0, 2, 1))  # (B, H*W, inner_dim)
        keys = self.key_proj(context)  # (B, N, inner_dim)
        values = self.value_proj(context)  # (B, N, inner_dim)

        # Reshape for multi-head attention
        queries = rearrange(queries, 'b n (h d) -> b h n d', h=self.num_heads)
        keys = rearrange(keys, 'b n (h d) -> b h n d', h=self.num_heads)
        values = rearrange(values, 'b n (h d) -> b h n d', h=self.num_heads)
        # Scaled dot-product attention
        attention_scores = torch.einsum('b h i d, b h j d -> b h i j', queries, keys) * self.scale
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_output = torch.einsum('b h i j, b h j d -> b h i d', attention_weights, values)

        # Reshape back to original dimensions
        attention_output = rearrange(attention_output, 'b h n d -> b n (h d)')
        attention_output = self.out_proj(attention_output)
        attention_output = attention_output.permute(0, 2, 1).view(B, C, H, W)
        
        return x + attention_output

class AttentionBlock(nn.Module):
    def __init__(self, in_channels, embed_dim, context_dim, num_heads=4, use_time_emb=False, dropout=0.0):
        super(AttentionBlock, self).__init__()
        self.use_time_emb = use_time_emb
        
        # Self-Attention
        self.self_attention_norm = Normalize(in_channels)
        self.self_attention = SelfAttention(in_channels, use_time_emb=use_time_emb)
        
        # Cross-Attention
        self.cross_attention_norm = Normalize(in_channels)
        self.cross_attention = CrossAttention(embed_dim=embed_dim, context_dim=context_dim, num_heads=num_heads)

        # Feedforward network
        self.ffn_norm = Normalize(in_channels)
        self.ffn = nn.Sequential(
            nn.Linear(in_channels, in_channels * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(in_channels * 4, in_channels),
            nn.Dropout(dropout)
        )
        
        # Activation
        self.silu = nn.SiLU()

    def forward(self, x, context, t=None):
        # Self-Attention with Residual Connection
        h = self.self_attention_norm(x)
        h = self.self_attention(h, t)
        x = x + h  # Residual Connection
        
        # Cross-Attention with Residual Connection
        h = self.cross_attention_norm(x)
        h = self.cross_attention(h, context)
        x = x + h  # Residual Connection
        
        # Feedforward network with Residual Connection
        h = self.ffn_norm(x)
        h = self.ffn(h)
        return x + h