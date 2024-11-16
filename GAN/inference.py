# inference.py
#%% Imports
import os, sys
import argparse
import importlib
import torch
from pytorch_fid import fid_score
from torchvision.utils import save_image

from modules.utils import set_random_seed
from evaluate.evaluate import convert2png, calculate_fid

# W&B 연결 설정
import subprocess
try:
    import wandb
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "wandb"])
    with open("../wandb_api.txt", "r") as f:
        key = f.readlines()
    subprocess.run(["wandb", "login"], input=key[0], encoding='utf-8')
    import wandb

#%% W&B 프로젝트 설정
project = "GAN"  # W&B 프로젝트 이름 설정
run = wandb.init(
    project=project, 
    tags=['inference'],  # 태그 설정
)

#%% Helper function to parse arguments
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
    parser.add_argument("--ver", type=int, default=0, help="fix the seed")
    
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--dataset", type=str, default="mnist")
    parser.add_argument("--epochs", type=int, default=1000, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=128, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension")
    parser.add_argument("--channels", type=int, default=1, help="number of image channels")
    parser.add_argument("--noise_size", default=100, type=int)

    parser.add_argument("--FID_size",type=int, default=1024)
    parser.add_argument("--dims", type=int, default=2048)
    
    if debug:
        return parser.parse_args(args=[])
    else:
        return parser.parse_args()

#%% Main function
def main():
    #%% configuration
    config = vars(get_args(debug=True))
    """ dataset """
    dataset_module = importlib.import_module('datasets.preprocess')
    importlib.reload(dataset_module)
    train_dataset, _, _, _ = dataset_module.get_mnist_dataloader(image_size=config["img_size"])
    #%%
    """model load"""
    base_name = f"{config['dataset']}_{config['lr']}_{config['batch_size']}"
    model_name = f"GAN_{base_name}"
    artifact = wandb.use_artifact(
        f"{project}/{model_name}:v{config['ver']}",
        type='model'
    )
    for key, item in artifact.metadata.items():
        config[key] = item
    model_dir = artifact.download()
    model_name = [x for x in os.listdir(model_dir) if x.endswith(f"{config['seed']}.pth")][0]
    
    config["cuda"] = torch.cuda.is_available()
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    wandb.config.update(config)
    
    set_random_seed(config["seed"])
    torch.manual_seed(config["seed"])
    #%%
    """ model load """
    gan_module = importlib.import_module('model.model')
    importlib.reload(gan_module)
    model = gan_module.GAN(config)
    model.to(device)
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
    print(f"Number of DISC Parameters: {discriminator_num_params/1_000}k")
    print(f"Number of GEN Parameters: {generator_num_params/1_000}k")
    wandb.log({"Number of DISC Parameters (k)": discriminator_num_params / 1_000})
    wandb.log({"Number of GEN Parameters (k)": generator_num_params / 1_000})
    wandb.log({"Total Number of GAN Parameters (k)": (discriminator_num_params + generator_num_params) / 1_000})
    #%%
    """ Image Generation """
    #%% 샘플 생성 및 저장
    num_samples = config['num_samples']
    generated_images = model.generate(num_samples=num_samples)

    output_dir = "./generated_samples"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for idx, img in enumerate(generated_images):
        file_name = f"{config['dataset']}_{idx+1:04d}.png"
        save_image(img, os.path.join(output_dir, file_name), normalize=True)
    wandb.log({"generated_images dir": output_dir})

    origin_png_dir = f"./{config['dataset']}_png_dir"
    convert2png(train_dataset, origin_png_dir)
    fid_score = calculate_fid(config, origin_png_dir, output_dir)
    print("fid_score >>> ",fid_score)
    wandb.log({"FID": fid_score})
    # Finish W&B run
    wandb.finish()
    print("Inference Done.")

#%% 실행
if __name__ == '__main__':
    main()
