#%%
import os
import torch
import argparse
import importlib
import torch
import matplotlib.pyplot as plt
from modules.utils import set_random_seed
from evaluate.evaluate import convert2png, fid_eval
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

project = "CVAE"
# entity = "hwawon" # put your WANDB username

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

    parser.add_argument("--num_samples", type=int, default=1000)

    parser.add_argument('--ver', type=int, default=0, 
                        help='model version number')
    
    parser.add_argument('--dataset', type=str, default='MNIST', 
                        help="""
                        Dataset options: MNIST, CIFAR10
                        """)
    parser.add_argument('--latent_dim', type=int, default=64, 
                        help='Dimension of latent space.')

    parser.add_argument("--FID_size",type=int, default=1024)
    parser.add_argument("--dims", type=int, default=2048)
    parser.add_argument('--num_classes', type = int, default = 10, help = 'number of classes')
    parser.add_argument('--encoded_info', default = {'channels':1, 'height':28, 'width':28},
                        help = 'channels, height, width')

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
    model_name = f"CVAE_{base_name}"
    artifact = wandb.use_artifact(
        f"{project}/{model_name}:v{config['ver']}",
        type='model')
    for key, item in artifact.metadata.items():
        config[key] = item
    model_dir = artifact.download()
    model_name = [x for x in os.listdir(model_dir) if x.endswith(f"{config['seed']}.pth")][0]
    #%%
    config["device"] = torch.cuda.is_available()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    config['device'] = device
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
    model = model_module.CVAE(config, config['encoded_info'], config['num_classes']).to(device)
    
    if config["device"]:
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
    # ax = model.generate(test_dataset)
    #%%
    """ Evaluate Score """
    #%%
    img_dir = './img_CVAE'
    wandb.log({"generated_images dir": img_dir})

    origin_png_dir = f"./{config['dataset']}_png_dir"
    convert2png(test_dataset, origin_png_dir)
    fid_value = fid_eval(config, origin_png_dir, img_dir)
    print("fid_score >>> ",fid_value)
    wandb.log({"FID": fid_value})
    
    # is_value = inception_score(config, img_dir)
    # print("inception-socre ")
    
#%% 
if __name__ == "__main__":
    main()  