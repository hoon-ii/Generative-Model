#%%
import argparse
import os, sys
import ast

import torch
import importlib
from collections import Counter

from modules.utils import set_seed
from modules.embedder import TextEmbedder

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

project="DDPM"

run = wandb.init(
    project=project, 
    # entity=entity, 
    tags=["train"],
)

#%%
def arg_as_list(s):
    v = ast.literal_eval(s)
    if type(v) is not list:
        raise argparse.ArgumentTypeError('Argument "%s" is not a list' % (s))
    return v

def get_args(debug=False):
    parser = argparse.ArgumentParser(description="VQVAE and UNet configurations")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--img_size", type=int, default=64)
    parser.add_argument("--in_dim", type=int, default=3)
    parser.add_argument("--h_dim", type=int, default=32)
    parser.add_argument("--res_h_dim", type=int, default=32)
    parser.add_argument("--n_res_layers", type=int, default=2)
    parser.add_argument("--n_embeddings", type=int, default=512)
    parser.add_argument("--beta", type=float, default=0.25)
    parser.add_argument("--outdim", type=int, default=3)
    
    parser.add_argument("--z_channels", type=int, default=8)
    parser.add_argument("--resolution", type=int, default=128)
    #Unet
    parser.add_argument("--embedding_dim", type=int, default=8)
    parser.add_argument("--base_ch", type=int, default=32)
    parser.add_argument("--ch_mult", nargs='+', type=int, default=[1, 2, 4])
    parser.add_argument("--attention_res", nargs='+', type=int, default=[32, 16])
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--num_groups", type=int, default=2)
    parser.add_argument("--cur_res", type=int, default=32)

    parser.add_argument("--out_ch", type=int, default=3)
    
    parser.add_argument("--num_res_blocks", type=int, default=2)
    parser.add_argument("--attn_resolutions", nargs='+', type=int, default=[128, 64])
    parser.add_argument("--dropout", type=float, default=0.0)

    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--batch_size", type=int, default=1)

    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=0.000048)
    parser.add_argument("--learning_rate2", type=float, default=0.000002)
    parser.add_argument("--loss", type=str, default="l2")

    parser.add_argument("--base_channels", type=int, default=32)
    
    parser.add_argument("--context_dim", type=int, default=512)
    

    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--beta_start", type=float, default=0.0001)
    parser.add_argument("--beta_end", type=float, default=0.02)
    parser.add_argument("--schedule", type=str, default="linear")
    parser.add_argument("--criterion", type=str, default="l2")
    parser.add_argument("--resamp_with_conv", type=bool , default=True)
    parser.add_argument("--attn_type", type=str , default='vanilla')
    parser.add_argument("--in_channels", type=int, default=3)

    parser.add_argument("--scheduler", type=str, default="linear")
    if debug:
        return parser.parse_args(argsa=[])
    else:
        return parser.parse_args()

def main():
    #%%
    config = vars(get_args(debug=False))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config['device'] = device 
    print('Current device is', device)
    set_seed(config["seed"])
    wandb.config.update(config)
    #%%
    """dataset"""
    dataset_module = importlib.import_module('datasets.preprocess')
    importlib.reload(dataset_module)
    train_dataset, _, mean, std= dataset_module.get_dataset(config["dataset"], config["img_size"])
    train_loader, _ = dataset_module.get_dataloader(train_dataset, _, config["dataset"], config['img_size'])
    # config["num_classes"] = len(Counter(train_dataset.targets.numpy()))
    #%%
    """ Model """
    diffusion_module = importlib.import_module('modules.model') #2개가 같이 있어야함.
    importlib.reload(diffusion_module)
    model = diffusion_module.DDPM(config)
    text_embedder = TextEmbedder(model_name="bert-base-uncased", output_dim=config['n_embeddings'])
    model.train().to(device)
    #%%
    """number of parameters"""
    count_parameters = lambda model: sum(p.numel() for p in model.parameters() if p.requires_grad)
    VQVAE_num_params = count_parameters(model.vqvae)
    UNET_num_params = count_parameters(model.unet)
    print(f"Number of VQVAE Parameters: {VQVAE_num_params/1_000_000}m")
    print(f"Number of UNET Parameters: {UNET_num_params/1_000_000}m")
    wandb.log({"Number of VQVAE Parameters (k)": VQVAE_num_params / 1_000_000})
    wandb.log({"Number of UNET Parameters (k)": UNET_num_params / 1_000_000})
    wandb.log({"Total Number of DDPM Parameters (k)": (VQVAE_num_params + UNET_num_params) / 1_000_000})
    #%%
    """ Train """
    train_module = importlib.import_module('modules.train')
    importlib.reload(train_module)
    v_optimizer = torch.optim.Adam(model.vqvae.parameters(), lr=config["learning_rate"])
    u_optimizer = torch.optim.Adam(model.unet.parameters(), lr=config["learning_rate2"])

    ''' 1st stage : VQVAE '''
    train_module.train_vqvae(
        config,
        model,
        v_optimizer, 
        train_loader
    )
    ''' 2dn stage : UNet '''
    train_module.train_unet(
        config,
        model,
        u_optimizer, 
        train_loader, 
        text_embedder,
    )
    #%%
    """ Model Save """
    base_name = f"{config['dataset']}__{config['batch_size']}"
    model_dir = f"./assets/models/{base_name}"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_name = f"DDPM_{base_name}_{config['seed']}"

    torch.save(model.state_dict(), f"./{model_dir}/{model_name}.pth")
    artifact = wandb.Artifact(
        "_".join(model_name.split("_")[:-1]), 
        type='model',
        metadata=config)

    artifact.add_file(f"./{model_dir}/{model_name}.pth")
    artifact.add_file('./main.py')
    artifact.add_file('./model/model.py')
    artifact.add_file('./modules/train.py')
    wandb.log_artifact(artifact)
    #%%
    wandb.config.update(config, allow_val_change=True)
    wandb.run.finish()
    print("W&B run finished.")

if __name__=="__main__":
    main()
