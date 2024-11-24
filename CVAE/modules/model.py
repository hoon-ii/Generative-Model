import torch
import torch.nn as nn
from torch.utils.data import DataLoader

#%% CVAE(fc layer로 구성)
class CVAE(nn.Module):
    def __init__(self, config, EncodedInfo, num_classes):
        super(CVAE, self).__init__()
        self.config = config
        self.channels = EncodedInfo['channels']
        self.height = EncodedInfo['height']
        self.width = EncodedInfo['width']
        self.input_dim = self.channels * self.height * self.width
        self.num_classes = num_classes  # 추가된 클래스 수 정보

        # Encoder
        en = []
        in_dim = self.input_dim + num_classes  # 조건을 포함한 입력 차원
        # print(f'channel {self.channels}, height {self.height}, width = {self.width}')
        # print(f'in dim {in_dim} = input dim {self.input_dim} + num_classes {num_classes}')

        for h in config["hidden_dims"]:
            en.append(nn.Linear(in_dim, h))
            en.append(nn.ReLU())
            in_dim = h
        en.append(nn.Linear(h, config["latent_dim"] * 2))  # mu, logvar 공간을 마련

        self.encoder = nn.Sequential(*en)

        # Decoder
        de = []
        in_dim = config["latent_dim"] + num_classes  # 조건을 포함한 latent 차원
        # print(f'decode in dim {in_dim} = latent dim {config["latent_dim"]} + num classes {num_classes}')

        for h in reversed(config["hidden_dims"]):
            de.append(nn.Linear(in_dim, h))
            de.append(nn.ReLU())
            in_dim = h
        de.append(nn.Linear(h, self.input_dim))
        de.append(nn.Sigmoid())

        self.decoder = nn.Sequential(*de)

    def encode(self, x, y):
        # 입력과 조건을 결합
        y_onehot = torch.nn.functional.one_hot(y, num_classes=self.num_classes).float()
        # print(f'encoded y onehot {y_onehot.shape}')
        # print(x.shape)
        x = x.view(-1, self.input_dim)  # Flatten input
        # print(x.shape)
        input_with_condition = torch.cat((x, y_onehot), dim=1)  # 결합
        mu_logvar = self.encoder(input_with_condition)
        return mu_logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, y):
        # Latent space와 조건 결합
        y_onehot = torch.nn.functional.one_hot(y, num_classes=self.num_classes).float()
        z_with_condition = torch.cat((z, y_onehot), dim=1)
        decoded = self.decoder(z_with_condition)
        decoded = decoded.view(-1, self.channels, self.height, self.width)
        return decoded

    def forward(self, x, y):
        mu_logvar = self.encode(x, y)
        mu, logvar = mu_logvar.chunk(2, dim=1)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, y)
        return recon, mu, logvar

    # def generate(self, y, num_samples, device):
    #     # 조건과 함께 샘플 생성
    #     z = torch.randn((num_samples, self.config["latent_dim"])).to(device)
    #     with torch.no_grad():
    #         generated_images = self.decode(z, y)
    #     return generated_images
    
    def generate(self, loader, labels, device):
        batch, _ = next(iter(loader))
        
        with torch.no_grad():
            batch = batch.to(device)
            labels = labels.to(device)
            generated_images, _, _ = self(batch, labels)  # Pass both data and labels
        return generated_images