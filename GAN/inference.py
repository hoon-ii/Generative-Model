# inference.py
#%% Imports
import os, sys
import argparse
import importlib
import torch
from torchvision.utils import save_image
from datasets.preprocess import get_mnist_dataloader

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
def arg_as_list(s):
    v = ast.literal_eval(s)
    if type(v) is not list:
        raise argparse.ArgumentTypeError('Argument "%s" is not a list' % (s))
    return v

def get_args(debug=False):
    parser = argparse.ArgumentParser("parameters")
    parser.add_argument("--ver", type=int, default=0, help="fix the seed")
    
    parser.add_argument("--dataset", type=str, default="mnist")
    parser.add_argument("--epochs", type=int, default=1000, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=128, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension")
    parser.add_argument("--channels", type=int, default=1, help="number of image channels")
    parser.add_argument("--noise_size", default=100, type=int)

    parser.add_argument("--hidden_size1", default=256, type=int)
    parser.add_argument("--hidden_size2", default=512, type=int)
    parser.add_argument("--hidden_size3", default=1024, type=int)

    if debug:
        return parser.parse_args(args=[])
    else:
        return parser.parse_args()

#%% Main function
def main():
    #%% configuration
    config = vars(get_args(debug=True))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config['device'] = device
    """model load"""
    base_name = f"{config['dataset']}_{config['lr']}_{config['batch_size']}"
    model_name_discriminator = f"discriminator_{base_name}_{config['seed']}.pth"
    model_name_generator = f"generator_{base_name}_{config['seed']}.pth"
    artifact_disc = wandb.use_artifact(
        f"{project}/{model_name_discriminator}:v{config['ver']}",
        type='model'
    )
    artifact_gen = wandb.use_artifact(
        f"{project}/{model_name_generator}:v{config['ver']}",
        type='model'
    )
    for key, item in artifact.metadata.items():
        config[key] = item
    model_dir = artifact.download()
    model_name_discriminator = [x for x in os.listdir(model_dir) if x.endswith(f"{config['seed']}.pth")][0]
    model_name_generator = [x for x in os.listdir(model_dir) if x.endswith(f"{config['seed']}.pth")][0]

    config["cuda"] = torch.cuda.is_available()
    device = 'cpu'
    # device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    wandb.config.update(config)
    
    set_random_seed(config["seed"])
    torch.manual_seed(config["seed"])
    #%% 모델 로드
    gan_module = importlib.import_module('model.model')
    importlib.reload(gan_module)
    generator = gan_module.Generator(config)
    generator.to(device)
    generator.eval()


    base_name = f"{config['dataset']}_{config['lr']}_{config['batch_size']}"
    model_dir = os.path.join("assets", "models", base_name)
    os.makedirs(model_dir, exist_ok=True)

    # File names and paths
    model_name_discriminator = f"discriminator_{base_name}_{config['seed']}.pth"
    model_name_generator = f"generator_{base_name}_{config['seed']}.pth"
    discriminator_path = os.path.join(model_dir, model_name_discriminator)
    generator_path = os.path.join(model_dir, model_name_generator)

    # Create W&B artifacts for both models
    artifact_disc = wandb.Artifact(
        f"{config['dataset']}_discriminator", 
        type='model',
        metadata=config
    )
    artifact_gen = wandb.Artifact(
        f"{config['dataset']}_generator", 
        type='model',
        metadata=config
    )

    # W&B artifact에서 모델 로드
    artifact = wandb.use_artifact(f"{project}/{config['ver']}", type="model")
    model_dir = artifact.download()
    generator_path = os.path.join(model_dir, f"generator_{base_name}_{config['ver']}.pth")
    generator.load_state_dict(torch.load(generator_path, map_location=device))
    
    #%% 샘플 생성 및 저장
    noise = torch.randn(config["num_samples"], config["noise_size"], device=device)
    with torch.no_grad():
        generated_images = generator(noise)
        generated_images = generated_images.view(config["num_samples"], 1, config["img_size"], config["img_size"])

    # 이미지 저장 및 W&B에 로그
    output_dir = "generated_samples"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "sample_images.png")
    save_image(generated_images, output_path, nrow=8, normalize=True)
    wandb.log({"generated_images": wandb.Image(output_path)})

    # Finish W&B run
    wandb.finish()
    print("Inference finished, samples saved and logged to W&B.")

#%% 실행
if __name__ == '__main__':
    main()
