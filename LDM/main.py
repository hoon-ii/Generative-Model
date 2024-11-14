#%% ==============================================================================
import os
import sys

import importlib
import argparse
import ast

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from modules.utils import set_random_seed

import subprocess
try:
    import wandb
except:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "wandb"])
    with open("./wandb_api.txt", "r") as f:
        key = f.readlines()
    subprocess.run(["wandb", "login"], input=key[0], encoding='utf-8')
    import wandb

project="LDM_final" # put your WANDB project name
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
    parser.add_argument('--seed', default=0, type=int, 
                        help="seed for repreatable results")
    parser.add_argument('--dataset', type=str, default='MNIST', 
                        help="""
                        datasets options: MNIST or CIFAR10""")
    
    ### Training setting
    parser.add_argument('--epochs', default=1000, type=int, 
                        help="Number of training epochs")
    parser.add_argument('--batch_size', default=64, type=int, 
                        help="Batch size for training")
    parser.add_argument('--lr', default=0.0001, type=float, 
                        help="Learning rate")
    parser.add_argument('--weight_decay', default=1e-3, type=float, 
                        help="Weight decay (L2 regularization)")
    
    ### VQVAE config
    parser.add_argument("--in_dim", default=3, type=int)
    parser.add_argument("--h_dim", default=4)
    parser.add_argument("--res_h_dim", default=32)
    parser.add_argument("--n_res_layers", default=2)
    parser.add_argument("--embedding_dim", default=8)
    parser.add_argument("--n_embeddings", default=16384)
    parser.add_argument("--beta", default=1)
    parser.add_argument("--save_img_embedding_map", default=False)
    parser.add_argument("--ch", default=128, type=int)
    parser.add_argument("--device", default='cpu', type=str)
    

    if debug:
        return parser.parse_args(args=[])
    else:
        return parser.parse_args()    

#%%
def main():
    #%% 
    """config"""
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
    CustomDataset = dataset_module.CustomDataset

    train_dataset = CustomDataset(
        config, 
        train=True
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        generator=torch.Generator(),
        # drop_last = True,
    )
    config['num_cont_features'] = len(train_dataset.continuous_features)
    #%% 
    """ Model(1) : VQVAE """
    vqvae_module = importlib.import_module('models.vqvae.vqvae')
    importlib.reload(vqvae_module)
    vqvae = vqvae_module.Model(config)
    vqvae.to(device)
    vqvae.train()
    """ Model(2) : UNet """
    unet_module = importlib.import_module('models.unet.unet')
    importlib.reload(unet_module)
    unet = unet_module.Model(config)
    unet.to(device)
    unet.train()
    #%%
    """number of parameters"""
    count_parameters = lambda model: sum(p.numel() for p in model.parameters() if p.requires_grad)
    vqvae_num_params = count_parameters(vqvae)
    unet_num_params = count_parameters(unet)
    print(f"Number of VQVAE Parameters: {vqvae_num_params/1_000_000}m")
    print(f"Number of UNET Parameters: {unet_num_params/1_000_000}m")
    wandb.log(f"Number of VQVAE Parameters: {vqvae_num_params+unet_num_params/1_000_000}millon")
    wandb.log(f"Number of UNET Parameters: {unet_num_params+unet_num_params/1_000_000}millon")
    wandb.log(f"Number of Model Parameters: {vqvae_num_params+unet_num_params/1_000_000}millon")
    #%%
    """ Train(1) : VQVAE """
    vqvae_train_module = importlib.import_module('modules.train_vqvae')
    importlib.reload(vqvae_train_module)
    optimizer = optim.Adam(
        vqvae.parameters(),
        lr=config["lr"],
        weight_decay=config['weight_decay']
    )
    vqvae_train_module.train_function(
        train_dataloader, 
        vqvae,
        optimizer, 
        config, 
        device,
    )
    """ Model(1) : VQVAE save"""
    base_name = f"{config['dataset']}_{config['lr']}_{config['batch_size']}_{config['alpha']}"
    model_dir = f"./assets/models/{base_name}"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_name = f"mywork_{base_name}_{config['seed']}"

    torch.save(vqvae.state_dict(), f"./{model_dir}/{model_name}.pth")
    artifact = wandb.Artifact(
        "_".join(model_name.split("_")[:-1]), 
        type='model',
        metadata=config)
    artifact.add_file(f"./{model_dir}/{model_name}.pth")
    artifact.add_file('./main.py')
    # artifact.add_file('./modules/model.py')
    # artifact.add_file('./modules/train.py')
    wandb.log_artifact(artifact)
    wandb.config.update(config, allow_val_change=True)
    #%%
    """ Train(2) : Unet """
    unet_train_module = importlib.import_module('modules.train_unet')
    importlib.reload(unet_train_module)
    optimizer = optim.Adam(
        unet.parameters(),
        lr=config["lr"],
        weight_decay=config['weight_decay']
    )
    unet_train_module.train_function(
        train_dataloader, 
        vqvae,
        optimizer, 
        config, 
        device,
    )
    """ Model(2) : Unet save"""
    base_name = f"{config['dataset']}_{config['lr']}_{config['batch_size']}_{config['alpha']}"
    model_dir = f"./assets/models/{base_name}"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_name = f"mywork_{base_name}_{config['seed']}"

    torch.save(unet.state_dict(), f"./{model_dir}/{model_name}.pth")
    artifact = wandb.Artifact(
        "_".join(model_name.split("_")[:-1]), 
        type='model',
        metadata=config)
    artifact.add_file(f"./{model_dir}/{model_name}.pth")
    artifact.add_file('./main.py')
    # artifact.add_file('./modules/model.py')
    # artifact.add_file('./modules/train.py')
    wandb.log_artifact(artifact)
    wandb.config.update(config, allow_val_change=True)
    #%%
    wandb.run.finish()

if __name__=='__main__':
    main()