import torch
import torch.nn as nn
import math
from modules.vqvae.moduels import Encoder, Decoder
from modules.vqvae.quantizer import VectorQuantizer


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device 
        half_dim = self.dim // 2
        embeddings = math.log(1000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings
    

class VQVAE(nn.Module):
    def __init__(self, config):
        super(VQVAE, self).__init__()
        
        self.embed_dim = config["embedding_dim"]
        self.n_embed = config["n_embeddings"]
        self.z_channels = config["z_channels"]
        self.save_img_embedding_map = config.get("save_img_embedding_map", False)  # 기본값 False
        
        encoder_config = {
            key: config[key] for key in [
                "base_ch", "in_channels", "resolution", "z_channels","out_ch", 
                "num_res_blocks", "attn_resolutions", "ch_mult", 
                 "dropout", "resamp_with_conv",
                 "attn_type"
            ]
        }
        decoder_config = {
            key: config[key] for key in [
                "base_ch", "in_channels", "resolution", "z_channels","out_ch", 
                "num_res_blocks", "attn_resolutions", "ch_mult", 
                 "dropout", "resamp_with_conv",
                 "attn_type"
            ]
        }

        self.encoder = Encoder(**encoder_config)
        self.decoder = Decoder(**decoder_config)
        
        self.vector_quantization = VectorQuantizer(
            self.n_embed, self.embed_dim, beta=config.get("vq_beta", 0.25)
        )
        
        self.pre_quantization_conv = nn.Conv2d(
            self.z_channels, self.embed_dim, kernel_size=1, stride=1
        )
        self.post_quantization_conv = nn.Conv2d(
            self.embed_dim, self.z_channels, kernel_size=1, stride=1
        )
        
        if self.save_img_embedding_map:
            self.img_to_embedding_map = {i: [] for i in range(self.n_embed)}
        else:
            self.img_to_embedding_map = None

    def encode(self, x):
        hidden = self.encoder(x)
        hidden = self.pre_quantization_conv(hidden)
        loss, z_q, perplexity, min_encodings, min_encoding_indices = self.vector_quantization(hidden)
        return loss, z_q, perplexity, min_encodings, min_encoding_indices
    
    def decode(self, quant):
        quant = self.post_quantization_conv(quant)
        dec = self.decoder(quant)
        return dec
    
    def forward(self, x, verbose=False):
        embed_loss, z_q, perplexity, min_encodings, min_encoding_indices = self.encode(x)
        x_hat = self.decode(z_q)

        if verbose:
            print('Original data shape:', x.shape)
            print('Reconstructed data shape:', x_hat.shape)
            assert False

        return embed_loss, x_hat, perplexity
    