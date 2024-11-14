#%%
import torch
from torch import nn
from torch.utils.data import DataLoader
import math
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
#%%
# convVAE(Conv2d layer로 구성)       
class convVAE(nn.Module):
    def __init__(self, config, EncodedInfo):
        super(convVAE, self).__init__()
        self.config = config
        self.channels = EncodedInfo.channels
        self.height = EncodedInfo.height
        self.width = EncodedInfo.width

        # Encoder
        en = []
        outputpaddings = []
        in_dim = self.channels
        self.red_h = self.height
        self.red_w = self.width

        for h in config["hidden_dim"]:
            en.append(
                nn.Sequential(
                    nn.Conv2d(in_channels = in_dim,
                              out_channels = h,
                              kernel_size = 3,
                              stride = 2,
                              padding = 1),
                    nn.BatchNorm2d(h),
                    nn.LeakyReLU()
                )
            )
            in_dim = h
            outputpaddings.append(0) if (self.red_h-1)%2 == 0 else outputpaddings.append(1)
            self.red_h = math.floor((self.red_h-1)/2) + 1
            self.red_w = math.floor((self.red_w-1)/2) + 1

        self.encoder = nn.Sequential(*en)
        # red는 Conv2d layer를 거쳐 축소된 크기
        # kernel = 3, stride = 2, padding = 1일 때 output의 크기는 2배씩 줄어듦
        # hidden dim의 개수만큼 Conv2d layer를 거치므로 2**(hidden dim 개수)만큼 줄어듦
        self.hidden_ch = h
        self.flat_dim = self.hidden_ch * self.red_h * self.red_w # 마지막 hidden dim 크기(채널 수) * 줄어든 feature map 크기(H*W)
        self.en_fc = nn.Linear(self.flat_dim, config["latent_dim"] * 2)
        
        # Decoder
        self.de_fc = nn.Linear(config["latent_dim"], self.flat_dim)
        de = []
        in_dim = self.hidden_ch
        for h, outpad in zip(reversed([self.channels]+config["hidden_dim"][:-1]), reversed(outputpaddings)):
            de.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        in_channels = in_dim,
                        out_channels = h,
                        kernel_size = 3,
                        stride = 2,
                        padding = 1,
                        output_padding= outpad    
                    ),
                    nn.BatchNorm2d(h),
                    nn.LeakyReLU()
                )
            )
            in_dim = h
        de.append(nn.Sigmoid())
        self.decoder = nn.Sequential(*de)
     
    def encode(self, x):
        encoded = self.encoder(x)
        encoded = encoded.view(-1, self.flat_dim) # en_fc에 넣기 위해 flatten
        mu_logvar = self.en_fc(encoded)
        return mu_logvar
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
    
    def decode(self, z):
        decoded = self.de_fc(z)
        decoded = decoded.view(-1, self.hidden_ch, self.red_h, self.red_w) # decode에 넣기 위해 reshape
        decoded = self.decoder(decoded)
        return decoded
    
    def forward(self, x):
        mu_logvar = self.encode(x)
        mu, logvar = mu_logvar.chunk(2, dim = 1)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar
    
    def generate(self, test_dataset, device):
        test_dataloader = DataLoader(
            test_dataset, 
            batch_size=64,
        )
        batch, label = next(iter(test_dataloader))
        
        with torch.no_grad():
            batch, label = batch.to(device), label.to(device)
            recon, mu, logvar = self(batch)
       
        grid = gridspec.GridSpec(3, 3)
        plt.figure(figsize = (10, 10))
        plt.subplots_adjust(wspace = 0.5, hspace = 0.5)
        for i in range(9):    
            ax = plt.subplot(grid[i])
            plt.imshow(recon[i].reshape(28, 28).cpu().detach().numpy(), cmap='gray')
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            ax.set_title('label : {}'.format(label[i]))

        return ax
            # %%
