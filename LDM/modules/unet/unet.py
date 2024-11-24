#%%
import torch
import torch.nn as nn

from modules.unet.attention import SelfAttention, CrossAttention, SinusoidalPositionEmbeddings, AttentionBlock

#%%
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()

        if not mid_channels:
            mid_channels = out_channels

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True)
        )
 
    def forward(self, x):
        return self.double_conv(x)
    

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1), # from Maxpool2D to AvgPool2d (09.05) > just conv2d
            DoubleConv(out_channels, out_channels)
        )

    def forward(self, x):
        return self.conv(x)
    

class Up(nn.Module):
    """Upscaling with ConvTranspose2d then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up_conv = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            DoubleConv(out_channels, out_channels)
        )

    def forward(self, x):
        return self.up_conv(x)
        

class UNetBlock(nn.Module):
    def __init__(self, in_channels,
                        out_channels, 
                        context_dim, 
                        num_heads, 
                        num_groups, 
                        use_attention, 
                        dropout # Create 09.12 
    ):
        super(UNetBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.block1 = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                    nn.GroupNorm(num_groups, out_channels),
                    nn.SiLU()
        )

        self.block2 = nn.Sequential(
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                    nn.GroupNorm(num_groups, out_channels),
        )
        self.dropout = torch.nn.Dropout(dropout)
        self.use_attention = use_attention
        if self.use_attention:
            self.attention_block = CrossAttention(out_channels, context_dim, num_heads)
        else:
            self.attention_block = None
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x, context=None, t=None):
        h = x
        h = self.block1(h)
        h = self.dropout(h)
        B, C, H, W = h.shape
        # condition with time if required : Sure!
        if t is not None:
            time_emb = SinusoidalPositionEmbeddings(H*W)(t)
            time_emb = time_emb.unsqueeze(1)  # Reshape for broadcasting to (b, dim, 1)
            time_emb = time_emb.repeat(1,C,1)
            time_emb = time_emb.view(h.shape)  # Reshape back to (b, c, h, w)
            h = h + time_emb  # Add time embedding to the input feature map

        h = self.block2(h)
        
        if self.use_attention == True and context is not None: 
            h = self.attention_block(h, context, t)
        return h + self.shortcut(x)
        

class UNet(nn.Module):
    def __init__(self, 
                 in_channels, 
                 base_channels,
                 ch_mult, 
                 attention_res,
                 context_dim, 
                 num_groups, 
                 num_heads,
                 cur_res,
                 dropout=0.0):
        super(UNet, self).__init__()

        cur_ch = in_channels
        self.cur_res = cur_res
        self.ch_mult = ch_mult
        # down sizing
        self.down_blocks = nn.ModuleList([])

        for i in range(len(ch_mult)):
            use_attention = cur_res in attention_res
            out_channels = base_channels * self.ch_mult[i]
            self.down_blocks.append(
                nn.ModuleList([
                    UNetBlock(cur_ch, out_channels, context_dim,
                              num_heads, num_groups, use_attention,
                              dropout=dropout
                                ).to('cuda'),
                    UNetBlock(out_channels, out_channels,context_dim,
                                num_heads, num_groups, use_attention,
                                dropout=dropout
                                ).to('cuda'),
                    nn.GroupNorm(1, out_channels),
                    CrossAttention(embed_dim=out_channels,
                                    context_dim=context_dim,
                                    num_heads=num_heads).to('cuda') if use_attention else nn.Identity(),
                    Down(in_channels=out_channels, 
                         out_channels=out_channels)
                ])
            )
            cur_res = cur_res // 2
            cur_ch = out_channels
            
        # Middle block
        self.mid_block_in = UNetBlock(in_channels=cur_ch,
                                      out_channels=cur_ch * 2,
                                      use_attention=use_attention,
                                      context_dim=context_dim, 
                                      num_heads=4, 
                                      num_groups=2, 
                                      dropout=dropout
        ).to('cuda')
        self.mid_sa = SelfAttention(cur_ch * 2, use_time_emb=True).to('cuda')
        self.mid_block_out = UNetBlock(in_channels=cur_ch * 2,
                                      out_channels=cur_ch * 2,
                                      use_attention=use_attention,
                                      context_dim=context_dim, 
                                      num_heads=4, 
                                      num_groups=2,
                                      dropout=dropout
        ).to('cuda')
        # up sizing
        self.up_blocks = nn.ModuleList([])
        for i in reversed(range(len(ch_mult))):
            use_attention = cur_res in attention_res
            in_ch = base_channels * self.ch_mult[i] * 2
            out_ch = base_channels * self.ch_mult[i]
            self.up_blocks.append(
                nn.ModuleList([
                    Up(in_channels=in_ch, out_channels=out_ch),
                    UNetBlock(in_ch, out_ch, context_dim,
                                num_heads, num_groups,use_attention,
                                dropout=dropout
                                ).to('cuda'),
                    UNetBlock(out_ch , out_ch, context_dim,
                                num_heads, num_groups, use_attention,
                                dropout=dropout
                                ).to('cuda'),
                    nn.GroupNorm(1, out_ch),
                    CrossAttention(embed_dim=out_ch,
                       context_dim=context_dim,
                       num_heads=num_heads).to('cuda') if use_attention else nn.Identity(),
                ])
            )

        self.final_conv = nn.Conv2d(out_ch, in_channels, 1).to('cuda')
        self.silu = nn.SiLU()

    def forward(self, x, context, t):
        # Time embed
        B, C, H, W = x.shape
        time_emb = SinusoidalPositionEmbeddings(H*W)(t)
        time_emb = time_emb.unsqueeze(1)  # Reshape for broadcasting to (b, dim, 1)
        time_emb = time_emb.repeat(1,C,1)
        time_emb = time_emb.view(x.shape)  # Reshape back to (b, c, h, w)
        x = x + time_emb  # Add time embedding to the input feature map
        x = nn.SiLU()(x)

        skips = []      
        for block1, block2, g_norm, attn, downsample in self.down_blocks.to('cuda'):
            x = block1(x, context, t)  
            x = block2(x, context, t)  
            x = g_norm(x)
            if isinstance(attn, nn.Identity):
                x = attn(x)  # `Identity`는 하나의 인자만 받습니다.
            else:
                x = attn(x, context, t)  # Attention이 필요한 경우 두 인자 모두 전달
            skips.append(x)
            x = downsample(x)
        x = self.mid_block_in(x, context, t)
        x = self.mid_sa(x, t)
        x = self.mid_block_out(x, context, t)

        for upsample, block1, block2, g_norm, attn in self.up_blocks.to('cuda'):
            x = upsample(x)
            x = torch.cat((x, skips.pop()), dim=1)
            x = block1(x, context, t)
            x = block2(x, context, t)
            x = g_norm(x)
            if isinstance(attn, nn.Identity):
                x = attn(x)  # `Identity`는 하나의 인자만 받습니다.
            else:
                x = attn(x, context, t)  # Attention이 필요한 경우 두 인자 모두 전달

        x = self.final_conv(x)
    
        return x

