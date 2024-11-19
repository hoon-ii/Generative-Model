#%%
import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, config):
        super(Discriminator, self).__init__()
        self.img_size = config["img_size"] ** 2
        self.hidden_size3 = config["hidden_size3"]
        self.hidden_size2 = config["hidden_size2"]
        self.hidden_size1 = config["hidden_size1"]
        
        self.linear1 = nn.Linear(self.img_size, self.hidden_size3)
        self.linear2 = nn.Linear(self.hidden_size3, self.hidden_size2)
        self.linear3 = nn.Linear(self.hidden_size2, self.hidden_size1)
        self.linear4 = nn.Linear(self.hidden_size1, 1)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.leaky_relu(self.linear1(x))
        x = self.leaky_relu(self.linear2(x))
        x = self.leaky_relu(self.linear3(x))
        x = self.linear4(x)
        x = self.sigmoid(x)
        return x

class Generator(nn.Module):
    def __init__(self, config):
        super(Generator, self).__init__()
        self.img_size = config["img_size"] ** 2
        self.hidden_size3 = config["hidden_size3"]
        self.hidden_size2 = config["hidden_size2"]
        self.hidden_size1 = config["hidden_size1"]
        self.noise_size = config["noise_size"]
        
        self.linear1 = nn.Linear(self.noise_size, self.hidden_size1)
        self.linear2 = nn.Linear(self.hidden_size1, self.hidden_size2)
        self.linear3 = nn.Linear(self.hidden_size2, self.hidden_size3)
        self.linear4 = nn.Linear(self.hidden_size3, self.img_size)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.relu(self.linear3(x))
        x = self.linear4(x)
        x = self.tanh(x)
        return x

class GAN(nn.Module):
    def __init__(self, config):
        super(GAN, self).__init__()
        self.config = config
        self.Discriminator = Discriminator(config)
        self.Generator = Generator(config)

    def disc_step(self, x):
        return self.Discriminator(x)
    
    def gen_step(self, x):
        return self.Generator(x)

    def generate(self, num_samples, init_noise=None):
        if init_noise is not None:
            noise = init_noise
        else:
            noise = torch.randn((num_samples, self.config["noise_size"]),
                                device=self.config["device"])
        generated_images = self.Generator(noise).view(
            num_samples,
            1,
            self.config["img_size"],
            self.config["img_size"])
        
        return generated_images