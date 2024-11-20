#%%
import os
import torch
import argparse
import importlib
from torchvision.utils import save_image
import torch
import matplotlib.pyplot as plt
from modules.utils import set_random_seed
from evaluate.evaluate import convert2png, evalutae

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

project = "VAE"
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
    parser.add_argument('--latent_dim', type=int, default=20, 
                        help='Dimension of latent space.')

    parser.add_argument("--FID_size",type=int, default=1024)
    parser.add_argument("--dims", type=int, default=2048)


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
    model_name = f"VAE_{base_name}"
    artifact = wandb.use_artifact(
        f"{project}/{model_name}:v{config['ver']}",
        type='model')
    for key, item in artifact.metadata.items():
        config[key] = item
    model_dir = artifact.download()
    model_name = [x for x in os.listdir(model_dir) if x.endswith(f"{config['seed']}.pth")][0]
    #%%
    config["cuda"] = torch.cuda.is_available()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

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
    model = model_module.VAE(config, train_dataset.EncodedInfo).to(device)
    
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
    # ax = model.generate(test_dataset)
    #%%
    """ Image Generation """
    #%%
    num_samples = config['num_samples']
    generated_images = model.generate(test_dataset, num_samples, device)

    output_dir = f"./generated_samples/{model_name}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for idx, img in enumerate(generated_images):
        file_name = f"{config['dataset']}_{idx+1:04d}.png"
        save_image(img, os.path.join(output_dir, file_name), normalize=True)
    wandb.log({"generated_images dir": output_dir})

    origin_png_dir = f"./{config['dataset']}_png_dir"
    convert2png(test_dataset, origin_png_dir)
    fid_score = evalutae(config, origin_png_dir, output_dir)
    print("fid_score >>> ",fid_score)
    wandb.log({"FID": fid_score})
#%% 
if __name__ == "__main__":
    main()  
