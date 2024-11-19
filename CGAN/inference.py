#%%
import os, sys
import argparse
import importlib
import torch
from torchvision.utils import save_image

from modules.utils import set_random_seed
from evaluate.evaluate import convert2png, evalutae

import subprocess
try:
    import wandb
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "wandb"])
    with open("../wandb_api.txt", "r") as f:
        key = f.readlines()
    subprocess.run(["wandb", "login"], input=key[0], encoding='utf-8')
    import wandb

#%%
project = "CGAN" 
run = wandb.init(
    project=project, 
    tags=['inference'], 
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

def get_args(debug=False):
    parser = argparse.ArgumentParser("parameters")
    parser.add_argument("--ver", type=int, default=0, help="model version number")
    
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--dataset", type=str, default="image dataset")
    parser.add_argument("--batch_size", type=int, default=128, help="size of the batches")
    parser.add_argument("--lr_d", type=float, default=0.0001, help="adam: learning rate")
    parser.add_argument("--lr_g", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension")
    parser.add_argument("--noise_size", type=int, default=512, help="noise size")

    parser.add_argument("--base_ch", type=list, default=256)
    parser.add_argument("--ch_mult", type=list, default=2, help="3 means that 1>2>4>8 times")

    parser.add_argument("--FID_size",type=int, default=1024)
    parser.add_argument("--dims", type=int, default=2048)
    
    if debug:
        return parser.parse_args(args=[])
    else:
        return parser.parse_args()

#%%
def main():
    #%%
    config = vars(get_args(debug=False))
    """model load"""
    base_name = f"{config['dataset']}_{config['lr_g']}_{config['lr_d']}_{config['batch_size']}"
    model_name = f"CGAN_{base_name}"
    artifact = wandb.use_artifact(
        f"{project}/{model_name}:v{config['ver']}",
        type='model'
    )
    print("Artifact Metadata:", artifact.metadata)
    for key, item in artifact.metadata.items():
        config[key] = item
    model_dir = artifact.download()
    model_name = [x for x in os.listdir(model_dir) if x.endswith(f"{config['seed']}.pth")][0]
    
    config["cuda"] = torch.cuda.is_available()
    print(f"Config after argparse parsing: {config}") 
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print('device is', device)
    wandb.config.update(config)
    
    set_random_seed(config["seed"])
    torch.manual_seed(config["seed"])
    #%%
    """ dataset """
    dataset_module = importlib.import_module('datasets.preprocess')
    importlib.reload(dataset_module)
    train_dataset, _, _, _ = dataset_module.get_mnist_dataloader(image_size=config["img_size"])
    #%%
    """ model load """
    gan_module = importlib.import_module('model.model')
    importlib.reload(gan_module)
    model = gan_module.GAN(config)
    model.to(device)
    model.eval()
    if config["cuda"]:
        model.load_state_dict(
            torch.load(
                model_dir + '/' + model_name
            )
        )
    else:
        model.load_state_dict(
            torch.load(
                model_dir + '/' + model_name, 
                map_location=torch.device('cpu'),
            )
        )
    model.eval()
    #%%
    """number of parameters"""
    count_parameters = lambda model: sum(p.numel() for p in model.parameters() if p.requires_grad)
    discriminator_num_params = count_parameters(model.Discriminator)
    generator_num_params = count_parameters(model.Generator)
    print(f"Number of DISC Parameters: {discriminator_num_params/1_000:.3f}k")
    print(f"Number of GEN Parameters: {generator_num_params/1_000:.3f}k")
    wandb.log({"Number of DISC Parameters (k)": discriminator_num_params / 1_000})
    wandb.log({"Number of GEN Parameters (k)": generator_num_params / 1_000})
    wandb.log({"Total Number of GAN Parameters (k)": (discriminator_num_params + generator_num_params) / 1_000})
    #%%
    """ Image Generation """
    #%%
    num_samples = config['num_samples']
    generated_images = model.generate(num_samples=num_samples)

    output_dir = f"./generated_samples/{model_name}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for idx, img in enumerate(generated_images):
        file_name = f"{config['dataset']}_{idx+1:04d}.png"
        save_image(img, os.path.join(output_dir, file_name), normalize=True)
    wandb.log({"generated_images dir": output_dir})

    origin_png_dir = f"./{config['dataset']}_png_dir"
    convert2png(train_dataset, origin_png_dir)
    fid_score = evalutae(config, origin_png_dir, output_dir)
    print("fid_score >>> ",fid_score)
    wandb.log({"FID": fid_score})
    #%%
    wandb.finish()
    print("Inference Done.")

#%% 실행
if __name__ == '__main__':
    main()
