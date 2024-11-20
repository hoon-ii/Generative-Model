#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 

class Discriminator(nn.Module):
    def __init__(self, config):
        super(Discriminator, self).__init__()
        self.config = config
        self.img_size = config["img_size"] ** 2
        self.hidden_sizes = [config["base_ch"] * (2**i) for i in range(config['ch_mult']+1)]
        self.num_classes = config["num_classes"]

        layers = []
        in_features = self.img_size + self.num_classes   
        for hidden_size in self.hidden_sizes:
            layers.append(nn.Linear(in_features, hidden_size))
            layers.append(nn.LeakyReLU(0.2))
            in_features = hidden_size   
        
        layers.append(nn.Linear(self.hidden_sizes[-1], 1))  
        layers.append(nn.Sigmoid())  
        
        self.model = nn.Sequential(*layers)

    def forward(self, img, target):
        target = F.one_hot(target, num_classes=self.config["num_classes"])
        d_input = torch.cat((img.view(img.size(0), -1), target), dim=1) 
        return self.model(d_input)

class Generator(nn.Module):
    def __init__(self, config):
        super(Generator, self).__init__()
        self.config = config
        self.img_size = config["img_size"] ** 2  
        self.noise_size = config["noise_size"]
        
        self.hidden_sizes = [config["base_ch"] * (2**i) for i in range(config['ch_mult']+1)]
        self.num_classes = config["num_classes"]
        
        # self.embedder = nn.Embedding(self.num_classes, self.num_classes)
        
        layers = []
        in_features = self.noise_size + self.num_classes   
        for hidden_size in self.hidden_sizes:
            layers.append(nn.Linear(in_features, hidden_size))
            layers.append(nn.LeakyReLU())
            in_features = hidden_size   
        
        layers.append(nn.Linear(self.hidden_sizes[-1], self.img_size))
        layers.append(nn.Tanh())   
        
        self.model = nn.Sequential(*layers)

    def forward(self, noise, target):
        target = F.one_hot(target, num_classes=self.config["num_classes"])
        gen_input = torch.cat((noise, target), dim=1)  
        return self.model(gen_input)


class CGAN(nn.Module):
    def __init__(self, config):
        super(CGAN, self).__init__()
        self.config = config
        self.discriminator = Discriminator(config)
        self.generator = Generator(config)

    def generate(self, num_samples, context=None, init_noise=None):
        if init_noise is not None:
            noise = init_noise
        else:
            noise = torch.randn((num_samples, self.config["noise_size"]),
                                device=self.config["device"])
        if context is not None:
            context = context
        else:
            context = torch.randint(
                0, self.config["num_classes"], (num_samples,),device=self.config["device"])
        generated_images = self.generator(noise, context).view(
            num_samples,
            1,
            self.config["img_size"],
            self.config["img_size"])
        
        return generated_images