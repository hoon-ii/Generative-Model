# %%
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
# %%
import importlib
import argparse

import torch

from synthcity.utils.serialization import load_from_file

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluation.simulation import set_random_seed
from evaluation.evaluation import evaluate
from evaluation import utility
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
    tags=["inference"], # put tags of this python project
)
# %%
def get_args(debug):
    parser = argparse.ArgumentParser("parameters")

    parser.add_argument('--ver', type=int, default=0, 
                        help='model version number')
    parser.add_argument("--model", type=str, default="TabDDPM")
    parser.add_argument('--dataset', type=str, default='banknote', 
                        help="""
                        Tabular dataset options: 
                        banknote, whitewine, breast, bankruptcy, musk, madelon
                        """)
    
    parser.add_argument("--embedding_dim", type=int, default=128, 
                        help="embedding dimension")
    parser.add_argument("--num_timesteps", type=int, default=1000, 
                        help="the number of timesteps")

    if debug:
        return parser.parse_args(args=[])
    else:
        return parser.parse_args()

#%%
def main():
    # %%
    config = vars(get_args(debug=False))  # default configuration

    """model load"""
    model_name = f"{config['model']}_{config['embedding_dim']}_{config['num_timesteps']}_{config['dataset']}"
    artifact = wandb.use_artifact(
        f"{project}/{model_name}:v{config['ver']}",
        type='model'
    )
    for key, item in artifact.metadata.items():
        config[key] = item

    model_dir = artifact.download()
    model_name = [x for x in os.listdir(model_dir) if x.endswith(f"{config['seed']}.pkl")][0]
    config["cuda"] = torch.cuda.is_available()
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    wandb.config.update(config)

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
    test_dataset = CustomDataset(
        config, train=False)
    #%%
    """model load"""
    model = load_from_file(
        model_dir + '/' + model_name
    )
    #%%
    """Number of parameters"""
    count_parameters = lambda model: sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    num_params = count_parameters(model.model)
    print(f"Number of Parameters: {num_params/ 1000:.2f}k")
    wandb.log({"Number of Parameters": num_params / 1000})
    # %%
    """synthetic dataset generation """
    n = len(train_dataset.raw_data)
    syndata = model.generate(n).dataframe()
    #%%
    """evaluation"""
    results = evaluate(syndata, train_dataset, test_dataset, config, device)
    for x, y in results._asdict().items():
        print(f"{x}: {y:.3f}")
        wandb.log({f"{x}": y})
    
    # print("Marginal Distribution...")
    # figs = utility.marginal_plot(train_dataset.raw_data, syndata, config, model_name)
    #%%
    wandb.config.update(config, allow_val_change=True)
    wandb.run.finish()
#%%
if __name__ == "__main__":
    main()
# %%
