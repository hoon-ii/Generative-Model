#%%
import os
import torch
import argparse
import importlib

import torch
import matplotlib.pyplot as plt
from modules.utils import set_random_seed
#%%
import sys
import subprocess
try:
    import wandb
except:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "wandb"])
    with open("../wandb_api.txt", "r") as f:
        key = f.readlines()
    subprocess.run(["wandb", "login"], input=key[0], encoding='utf-8')
    import wandb

project = "convVAE" # put your WANDB project name
# entity = "wotjd1410" # put your WANDB username

run = wandb.init(
    project=project, 
    # entity=entity, 
    tags=["inference"], # put tags of this python project
)
# %%
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
def get_args(debug=False):
    parser = argparse.ArgumentParser('parameters')
    
    parser.add_argument('--ver', type=int, default=0, 
                        help='model version number')
    
    parser.add_argument('--dataset', type=str, default='MNIST', 
                        help="""
                        Dataset options: MNIST, CelebA
                        """)
    parser.add_argument('--latent_dim', type=int, default=128, 
                        help='Dimension of latent space.')
    if debug:
        return parser.parse_args(args=[])
    else:    
        return parser.parse_args()
    
#%%
def main():
    #%%
    config = vars(get_args(debug=True))
    #%%
    """model load"""
    base_name = f"{config['dataset']}_{config['latent_dim']}" 
    model_name = f"convVAE_{base_name}"
    artifact = wandb.use_artifact(
        f"{project}/{model_name}:v{config['ver']}",
        type='model')
    for key, item in artifact.metadata.items():
        config[key] = item
    model_dir = artifact.download()
    model_name = [x for x in os.listdir(model_dir) if x.endswith(f"{config['seed']}.pth")][0]
    #%%
    config["cuda"] = torch.cuda.is_available()
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    set_random_seed(config["seed"])
    wandb.config.update(config)
    #%%
    """dataset"""
    dataset_module = importlib.import_module('datasets.preprocess')
    importlib.reload(dataset_module)
    train_dataset = dataset_module.CustomMNIST(train=True)
    test_dataset = dataset_module.CustomMNIST(train=False)
    
    #%%
    """model"""
    model_module = importlib.import_module('modules.model')
    importlib.reload(model_module)
    model = model_module.convVAE(config, train_dataset.EncodedInfo).to(device)
    
    if config["cuda"]:
        model.load_state_dict(
            torch.load(
                model_dir + "/" + model_name
            )
        )
    else:
        model.load_state_dict(
            torch.load(
                model_dir + "/" + model_name,
                map_location=torch.device("cpu"),
            )
        )
    model.eval()
    #%%
    """number of model parameters"""
    count_parameters = lambda model: sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_params = count_parameters(model)
    print(f"Number of Parameters: {model_params/1000000:.1f}M")
    wandb.log({"Number of Parameters": model_params /1000000})
    #%%
    """generation"""
    ax = model.generate(test_dataset, device)
    #%%
    """image save"""
    base_name = f"{config['dataset']}_{config['latent_dim']}"
    figure_dir = f"./assets/figures/{base_name}"
    if not os.path.exists(figure_dir):
        os.makedirs(figure_dir)
    figure_name = f"convVAE_{base_name}_{config['seed']}"

    ax.figure.savefig(f"./{figure_dir}/{figure_name}.png")
    wandb.log({"figure" : ax})
    #%%
    wandb.config.update(config, allow_val_change=True)
    wandb.run.finish()
#%% 
if __name__ == "__main__":
    main()  