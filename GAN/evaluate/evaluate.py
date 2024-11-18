#%%
import os
import torch
import torchvision.transforms as transforms
from pytorch_fid import fid_score
from modules.utils import denormalize
def convert2png(dataset, origin_png_dir):
    if not os.path.exists(origin_png_dir):
        os.makedirs(origin_png_dir)

    for idx, (img, _) in enumerate(dataset):
        if isinstance(img, torch.Tensor):  # Tensor인지 확인
            img = denormalize(img)
            img = transforms.ToPILImage()(img)
        img_path = os.path.join(origin_png_dir, f"origin_{idx+1:04d}.png")
        img.save(img_path)
    
    print(f"{len(os.listdir(origin_png_dir))} images are done.")

def evalutae(config, origin_png_dir, output_dir):
    fid_value = fid_score.calculate_fid_given_paths(
        [origin_png_dir, output_dir],
        batch_size=config["FID_size"],
        device=config['device'],
        dims=config["dims"]
    )
    return fid_value
