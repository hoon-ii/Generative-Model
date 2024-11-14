#%%
import argparse
import os, sys
import numpy as np
import math
import ast

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import torch.optim as optim

import torch.nn as nn
import torch.nn.functional as F
import torch
import importlib

from modules.utils import set_random_seed
from datasets.preprocess import get_mnist_dataloader
import wandb

import subprocess
try:
    import wandb
except:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "wandb"])
    with open("./wandb_api.txt", "r") as f:
        key = f.readlines()
    subprocess.run(["wandb", "login"], input=key[0], encoding='utf-8')
    import wandb

project="GAN" # put your WANDB project name
# entity = "shoon06" # put your WANDB username

run = wandb.init(
    project=project, 
    # entity=entity, 
    tags=["train"], # put tags of this python project
)

#%%
def arg_as_list(s):
    v = ast.literal_eval(s)
    if type(v) is not list:
        raise argparse.ArgumentTypeError('Argument "%s" is not a list' % (s))
    return v

def get_args(debug=False):
    parser = argparse.ArgumentParser("parameters")
    parser.add_argument("--seed", type=int, default=0, help="fix the seed")
    
    parser.add_argument("--dataset", type=str, default="mnist")
    parser.add_argument("--epochs", type=int, default=1000, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=128, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension")
    parser.add_argument("--channels", type=int, default=1, help="number of image channels")
    parser.add_argument("--noise_size", default=100, type=int)

    parser.add_argument("--hidden_size1", default=256, type=int)
    parser.add_argument("--hidden_size2", default=512, type=int)
    parser.add_argument("--hidden_size3", default=1024, type=int)

    if debug:
        return parser.parse_args(argsa=[])
    else:
        return parser.parse_args()

def main():
    #%% configuration
    config = vars(get_args(debug=False))  # default configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config['device'] = device 
    print('Current device is', device)
    set_random_seed(config["seed"])
    wandb.config.update(config)
    #%%
    """dataset"""
    dataset_module = importlib.import_module('datasets.preprocess')
    importlib.reload(dataset_module)
    train_loader, test_loader = dataset_module.get_mnist_dataloader(config["batch_size"], config["img_size"])
    #%%
    """ Model """
    gan_module = importlib.import_module('model.model')
    importlib.reload(gan_module)
    discriminator = gan_module.Discriminator(config)
    generator = gan_module.Generator(config)
    discriminator.to(device)
    generator.to(device)
    #%%
    """number of parameters"""
    count_parameters = lambda model: sum(p.numel() for p in model.parameters() if p.requires_grad)
    discriminator_num_params = count_parameters(discriminator)
    generator_num_params = count_parameters(generator)
    print(f"Number of DISC Parameters: {discriminator_num_params/1_000}k")
    print(f"Number of GEN Parameters: {generator_num_params/1_000}k")
    wandb.log({"Number of VQVAE Parameters (k)": discriminator_num_params / 1_000})
    wandb.log({"Number of UNET Parameters (k)": generator_num_params / 1_000})
    wandb.log({"Total Number of Model Parameters (k)": (discriminator_num_params + generator_num_params) / 1_000})
    #%%
    """ Train """
    GAN_train_module = importlib.import_module('modules.train')
    importlib.reload(GAN_train_module)
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=config["lr"])
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=config["lr"])

    GAN_train_module.train(
        train_loader, 
        generator, 
        discriminator, 
        g_optimizer, 
        d_optimizer, 
        config, 
        device
    )
    #%%
    """ Model Save : Saves both Generator and Discriminator """
    # Directory for saving models
    base_name = f"{config['dataset']}_{config['lr']}_{config['batch_size']}"
    model_dir = os.path.join("assets", "models", base_name)
    os.makedirs(model_dir, exist_ok=True)

    # File names and paths
    model_name_discriminator = f"discriminator_{base_name}_{config['seed']}.pth"
    model_name_generator = f"generator_{base_name}_{config['seed']}.pth"
    discriminator_path = os.path.join(model_dir, model_name_discriminator)
    generator_path = os.path.join(model_dir, model_name_generator)

    # Save model state dictionaries
    torch.save(discriminator.state_dict(), discriminator_path)
    torch.save(generator.state_dict(), generator_path)
    print(f"Discriminator saved to {discriminator_path}")
    print(f"Generator saved to {generator_path}")

    # Create W&B artifacts for both models
    artifact_disc = wandb.Artifact(
        f"{config['dataset']}_discriminator", 
        type='model',
        metadata=config
    )
    artifact_gen = wandb.Artifact(
        f"{config['dataset']}_generator", 
        type='model',
        metadata=config
    )

    # Add model files and main script to artifacts
    artifact_disc.add_file(discriminator_path)
    artifact_gen.add_file(generator_path)
    artifact_disc.add_file('./main.py')
    artifact_gen.add_file('./main.py')
    
    # Log artifacts to W&B
    wandb.log_artifact(artifact_disc)
    wandb.log_artifact(artifact_gen)
    wandb.config.update(config, allow_val_change=True)

    # Finish W&B run
    wandb.finish()
    print("W&B run finished.")

if __name__=="__main__":
    main()