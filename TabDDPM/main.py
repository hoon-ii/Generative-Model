# %%
"""
[1] Reference: https://github.com/vanderschaarlab/synthcity.git

conda create --name TabDDPM python==3.9.7 
pip install synthcity
"""
# %%
import os
# os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import sys
import importlib

import torch

from synthcity.utils.serialization import save_to_file
from synthcity.plugins import Plugins

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluation.simulation import set_random_seed
# %%
import subprocess
try:
    import wandb
except:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "wandb"])
    with open("./wandb_api.txt", "r") as f:
        key = f.readlines()
    subprocess.run(["wandb", "login"], input=key[0], encoding="utf-8")
    import wandb

project = "distvae_journal_baseline1" # put your WANDB project name
entity = "anseunghwan" # put your WANDB username

run = wandb.init(
    project=project, 
    entity=entity, 
    # tags=[""], # put tags of this python project
)
# %%
import argparse
import ast

def arg_as_list(s):
    v = ast.literal_eval(s)
    if type(v) is not list:
        raise argparse.ArgumentTypeError('Argument "%s" is not a list' % (s))
    return v

def get_args(debug):
    parser = argparse.ArgumentParser("parameters")

    parser.add_argument("--seed", type=int, default=0, 
                        help="seed for repeatable results")
    parser.add_argument("--model", type=str, default="TabDDPM")
    parser.add_argument('--dataset', type=str, default='banknote', 
                        help="""
                        Tabular dataset options: 
                        banknote, whitewine, breast, bankruptcy, musk, madelon
                        """)
    
    parser.add_argument("--test_size", default=0.2, type=float,
                        help="the ratio of train test split")
    
    parser.add_argument("--embedding_dim", type=int, default=128, 
                        help="embedding dimension")
    parser.add_argument("--num_timesteps", type=int, default=1000, 
                        help="the number of timesteps")
    parser.add_argument("--n_iter", type=int, default=10000, 
                        help="the number of iterations")

    if debug:
        return parser.parse_args(args=[])
    else:
        return parser.parse_args()

# %%
def main():
    # %%
    config = vars(get_args(debug=False))  # default configuration
    config["cuda"] = torch.cuda.is_available()
    device = (
        torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    )
    wandb.config.update(config)
    print(device)

    set_random_seed(config["seed"])
    torch.manual_seed(config["seed"])
    if config["cuda"]:
        torch.cuda.manual_seed(config["seed"])
    # %%
    """dataset"""
    dataset_module = importlib.import_module('datasets.preprocess')
    importlib.reload(dataset_module)
    CustomDataset = dataset_module.CustomDataset

    train_dataset = CustomDataset(
        config, train=True)
    # %%
    """model"""
    model = Plugins().get(
        "ddpm", 
        random_state=config["seed"], 
        n_iter=config["n_iter"],
        dim_embed=config["embedding_dim"],
        num_timesteps=config["num_timesteps"],
        device=device
    )
    # %%
    """training"""
    model.fit(train_dataset.data)
    # %%
    """number of parameters"""
    count_parameters = lambda model: sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    num_params = count_parameters(model.model)
    print(f"Number of Parameters: {num_params/ 1000:.2f}k")
    #%%
    """model save"""
    base_name = f"{config['model']}_{config['embedding_dim']}_{config['num_timesteps']}_{config['n_iter']}_{config['dataset']}"
    model_dir = f"./assets/models/{base_name}/"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_name = f"{base_name}_{config['seed']}"
    save_to_file(f"./{model_dir}/{model_name}.pkl", model)
    artifact = wandb.Artifact(
        "_".join(model_name.split("_")[:-1]), 
        type='model',
        metadata=config) 
    artifact.add_file(f"./{model_dir}/{model_name}.pkl")
    artifact.add_file('./main.py')
    artifact.add_file('./datasets/preprocess.py')
    # artifact.add_file('./modules/train.py')
    # artifact.add_file('./modules/model.py')
    wandb.log_artifact(artifact)
    #%%
    wandb.config.update(config, allow_val_change=True)
    wandb.run.finish()
# %%
if __name__ == "__main__":
    main()
# %%
