#%%
import torch
import torch.nn as nn
import numpy as np
from models.vqvae.modules.encoder import Encoder
from models.vqvae.modules.quantizer import VectorQuantizer
from models.vqvae.modules.decoder import Decoder


class VQVAE(nn.Module):
    def __init__(self, config):
        super(VQVAE, self).__init__()
        # encode image into continuous latent space
        self.in_dim = config["in_dim"] #color for 3
        self.h_dim = config["h_dim"]
        self.res_h_dim = config["res_h_dim"]
        self.n_res_layers = config["n_res_layers"]
        self.n_embeddings = config["n_embeddings"]
        self.embedding_dim = config["embedding_dim"]
        self.beta = config["beta"]
        self.save_img_embedding_map = config["save_img_embedding_map"]
        self.device = config["device"]

        self.encoder = Encoder(self.in_dim, self.h_dim, self.n_res_layers, self.res_h_dim)
        self.pre_quantization_conv = nn.Conv2d(
            self.h_dim, self.embedding_dim, kernel_size=1, stride=1)
        # pass continuous latent vector through discretization bottleneck
        self.vector_quantization = VectorQuantizer(
            self.n_embeddings, self.embedding_dim, self.beta, self.device)
        # decode the discrete latent representation
        self.decoder = Decoder(self.embedding_dim, self.h_dim, self.n_res_layers, self.res_h_dim)
        
        if self.save_img_embedding_map:
            self.img_to_embedding_map = {i: [] for i in range(self.n_embeddings)}
        else:
            self.img_to_embedding_map = None

    def forward(self, x, verbose=False):
        z_e = self.encoder(x)
        z_e = self.pre_quantization_conv(z_e)
        embedding_loss, z_q, perplexity, _, _ = self.vector_quantization(
            z_e)
        x_hat = self.decoder(z_q)

        if verbose:
            print('original data shape:', x.shape)
            print('encoded data shape:', z_e.shape)
            print('recon data shape:', x_hat.shape)
            assert False

        return embedding_loss, x_hat, perplexity