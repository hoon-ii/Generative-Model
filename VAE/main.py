#%%
import os
import argparse
import importlib
#%%
import torch
from torch.utils.data import DataLoader
import torch.optim as optim

from modules.utils import set_random_seed
#%%
import sys
import subprocess
try:
    import wandb
except:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "wandb"])
    with open("./wandb_api.txt", "r") as f:
        key = f.readlines()
    subprocess.run(["wandb", "login"], input=key[0], encoding='utf-8')
    import wandb

project = "convVAE" # put your WANDB project name
# entity = "wotjd1410" # put your WANDB username

run = wandb.init(
    project=project, 
    # entity=entity, 
    tags=["train"], # put tags of this python project
)

#%%
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
#%%
def get_args(debug):
    parser = argparse.ArgumentParser('parameters')
    parser.add_argument("--seed", default=0, type=int,
                        help="seed for repeatable results") 
    
    parser.add_argument('--dataset', type=str, default='MNIST', 
                        help="""
                        Dataset options: MNIST, CelebA
                        """)
    parser.add_argument('--hidden_dim', type=list, default=[32, 64, 128], 
                        help='Number of input channels.')
    parser.add_argument('--latent_dim', type=int, default=128, 
                        help='Dimension of latent space.')

    # Data parameters
    parser.add_argument('--batch_size', type=int, default=64, 
                        help='Training batch size.')

    # Experiment parameters
    parser.add_argument('--lr', type=float, default=0.005, 
                        help='Learning rate.')
    parser.add_argument('--weight_decay', type=float, default=0.0, 
                        help='Weight decay.')
    parser.add_argument('--scheduler_gamma', type=float, default=0.95, 
                        help='Scheduler gamma.')
    parser.add_argument('--kld_weight', type=float, default=0.00025, 
                        help='Weight for the KL divergence loss.')
    # Trainer parameters
    parser.add_argument('--epochs', type=int, default=100, 
                        help='Maximum number of training epochs.')
    
    if debug:
        return parser.parse_args(args=[])
    else:
        return parser.parse_args()

#%%
def main():
    #%%
    config = vars(get_args(debug=True)) # default configuration
    set_random_seed(config['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Current device is', device)
    wandb.config.update(config)
    #%%
    """dataset"""
    dataset_module = importlib.import_module('datasets.preprocess')
    importlib.reload(dataset_module)
    train_dataset = dataset_module.CustomMNIST(train=True)
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=config["batch_size"]
    )
    #%%
    """model"""
    model_module = importlib.import_module('modules.model')
    importlib.reload(model_module)
    model = model_module.convVAE(config, train_dataset.EncodedInfo).to(device)
    #%%
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])
    #%%
    """number of model parameters"""
    count_parameters = lambda model: sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_params = count_parameters(model)
    print(f"Number of Parameters: {model_params/1000000:.1f}M")
    #%%
    """train"""
    train_module = importlib.import_module('modules.train')
    importlib.reload(train_module)
    train_module.train_function(
        model, 
        config, 
        optimizer,
        train_dataloader, 
        device
    )
    #%%
    """model save"""
    base_name = f"{config['dataset']}_{config['latent_dim']}"
    model_dir = f"./assets/models/{base_name}"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_name = f"convVAE_{base_name}_{config['seed']}"

    torch.save(model.state_dict(), f"./{model_dir}/{model_name}.pth")
    artifact = wandb.Artifact(
        "_".join(model_name.split("_")[:-1]), 
        type='model',
        metadata=config) 
    artifact.add_file(f"./{model_dir}/{model_name}.pth")
    artifact.add_file('./main.py')
    artifact.add_file('./modules/model.py')
    artifact.add_file('./modules/train.py')
    wandb.log_artifact(artifact)
    #%%
    wandb.config.update(config, allow_val_change=True)
    wandb.run.finish()
#%%
if __name__ == '__main__':
    main()