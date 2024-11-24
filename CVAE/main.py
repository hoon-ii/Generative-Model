import argparse
import torch
from torchvision import datasets, transforms
from torchvision.utils import save_image, make_grid
from torch.utils.data import DataLoader
import yaml
import wandb
import os
from modules.model import CVAE  # CVAE 모델이 저장된 파일을 'cvae_model.py'로 가정
from modules.utils import set_random_seed
import importlib
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

project = 'CVAE'
# entity = "hwawon" # put your WANDB username

run = wandb.init(
    project=project, 
    # entity=entity, 
    tags=["train"], # put tags of this python project
)

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_args(debug):
    parser = argparse.ArgumentParser('parameters')
    parser.add_argument("--seed", default=0, type=int,
                        help="seed for repeatable results") 
    
    parser.add_argument('--dataset', type=str, default='MNIST', 
                        help="""
                        Dataset options: MNIST, CIFAR10
                        """)
    parser.add_argument('--hidden_dims', type=list, default=[512, 256, 128], 
                        help="Number of neurons")
    parser.add_argument('--latent_dim', type=int, default=64, 
                        help='Dimension of latent space.')

    # Data parameters
    parser.add_argument('--batch_size', type=int, default=64, 
                        help='Training batch size.')
    parser.add_argument('--num_classes', type = int, default = 10, help = 'number of classes')

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
    parser.add_argument('--epochs', type=int, default=30, 
                        help='Maximum number of training epochs.')
    parser.add_argument('--encoded_info', default = {'channels':1, 'height':28, 'width':28},
                        help = 'channels, height, width')
    if debug:
        return parser.parse_args(args=[])
    else:
        return parser.parse_args()   

def main():
    config = vars(get_args(debug = False))
    set_random_seed(config['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Current device is', device)
    wandb.config.update(config)

    'dataset'
    dataset_module = importlib.import_module('datasets.preprocess')
    importlib.reload(dataset_module)

    if config['dataset'] == 'MNIST':
        train_dataset = dataset_module.CustomMNIST(train=True)
        test_dataset = dataset_module.CustomMNIST(train = False)
    elif config['dataset'] == 'CIFAR10':
        train_dataset = dataset_module.CustomCIFAR10(train = True)
        test_dataset = dataset_module.CustomCIFAR10(train = False)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size = config['batch_size'],
        shuffle = True
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size = 100,
        shuffle = False
    )

    # Model, loss function, and optimizer
    model_module = importlib.import_module('modules.model')
    importlib.reload(model_module)

    
    model = model_module.CVAE(config, config['encoded_info'], config['num_classes']).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])

    count_parameters = lambda model: sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_params = count_parameters(model)

    # Training loop
    train_module = importlib.import_module('modules.train')
    importlib.reload(train_module)
    # train_module.train_function(
    #     model,
    #     config,
    #     optimizer,
    #     train_dataloader,
    #     device
    # )

    train_module.train_function(
        model,
        config,
        optimizer,
        train_dataloader,
        test_dataloader,
        device
    )

   
    """model save"""
    base_name = f"{config['dataset']}_{config['latent_dim']}" # MNIST_64
    print(base_name)
    model_dir = f"./assets/models/{base_name}" # ./assets/models/MNIST_64
    print(model_dir)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_name = f"CVAE_{base_name}_{config['seed']}.pth" # CVAE_MNIST_64_0.pth
    print(model_name)
    save_path = os.path.join(model_dir, model_name) # assets/models/MNIST_64/CVAE_MNIST_64_0.pth
    print(save_path)
    torch.save(model.state_dict(), save_path)
    
    print("_".join(model_name.split("_")[:-1]))
    artifact = wandb.Artifact(
        "_".join(model_name.split("_")[:-1]), # CVAE_MNIST_64
        type='model',
        metadata=config) 
    print(f"{model_dir}/{model_name}")
    artifact.add_file(f"./{model_dir}/{model_name}") 
    artifact.add_file('./main.py')
    artifact.add_file('./modules/model.py')
    artifact.add_file('./modules/train.py') 
    wandb.log_artifact(artifact)
    #%%
    wandb.config.update(config, allow_val_change=True)
    wandb.run.finish()
    print("W&B run finished.")
#%%
if __name__ == '__main__':
    main()