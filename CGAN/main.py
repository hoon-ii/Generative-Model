#%%
import argparse
import os, sys
import ast

import torch
import importlib
from collections import Counter

from modules.utils import set_random_seed
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

project="CGAN"

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
    parser = argparse.ArgumentParser("parameters")
    parser.add_argument("--seed", type=int, default=0, help="fix the seed")
    
    parser.add_argument("--dataset", type=str, default="mnist")
    parser.add_argument("--epochs", type=int, default=300, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--lr_d", type=float, default=0.000005, help="adam: learning rate")
    parser.add_argument("--lr_g", type=float, default=0.00001, help="adam: learning rate")
    parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension")
    parser.add_argument("--channels", type=int, default=1, help="number of image channels")
    parser.add_argument("--noise_size", default=128, type=int)

    parser.add_argument("--base_ch", type=int, default=256)
    parser.add_argument("--ch_mult", type=int, default=2, help="3 means that 1>2>4>8 times")

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
    set_random_seed(config["seed"])
    wandb.config.update(config)
    #%%
    """dataset"""
    dataset_module = importlib.import_module('datasets.preprocess')
    importlib.reload(dataset_module)
    train_dataset, _, train_loader, _ = dataset_module.get_mnist_dataloader(config["batch_size"], config["img_size"])
    config["num_classes"] = len(Counter(train_dataset.targets.numpy()))
    #%%
    """ Model """
    cgan_module = importlib.import_module('model.model')
    importlib.reload(cgan_module)
    model = cgan_module.CGAN(config)
    model.train().to(device)
    #%%
    """number of parameters"""
    count_parameters = lambda model: sum(p.numel() for p in model.parameters() if p.requires_grad)
    discriminator_num_params = count_parameters(model.discriminator)
    generator_num_params = count_parameters(model.generator)
    print(f"Number of DISC Parameters: {discriminator_num_params/1_000}k")
    print(f"Number of GEN Parameters: {generator_num_params/1_000}k")
    wandb.log({"Number of DISC Parameters (k)": discriminator_num_params / 1_000})
    wandb.log({"Number of GEN Parameters (k)": generator_num_params / 1_000})
    wandb.log({"Total Number of CGAN Parameters (k)": (discriminator_num_params + generator_num_params) / 1_000})
    #%%
    """ Train """
    CGAN_train_module = importlib.import_module('modules.train')
    importlib.reload(CGAN_train_module)
    d_optimizer = torch.optim.Adam(model.discriminator.parameters(), lr=config["lr_d"])
    g_optimizer = torch.optim.Adam(model.generator.parameters(), lr=config["lr_g"])

    CGAN_train_module.train(
        train_loader, 
        model,
        g_optimizer, 
        d_optimizer, 
        config, 
        device
    )
    #%%
    """ Model Save """
    base_name = f"{config['dataset']}_{config['lr_g']}_{config['lr_d']}_{config['batch_size']}"
    model_dir = f"./assets/models/{base_name}"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_name = f"CGAN_{base_name}_{config['seed']}"

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
