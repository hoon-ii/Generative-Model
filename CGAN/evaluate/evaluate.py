#%%
import os
import torch
from torchvision.models import inception_v3
from torch.nn.functional import softmax
from pytorch_fid import fid_score
from PIL import Image
import numpy as np
from scipy.stats import entropy
import torchvision.transforms as transforms
from pytorch_fid import fid_score
from modules.utils import denormalize


def convert2png(dataset, origin_png_dir):
    if not os.path.exists(origin_png_dir):
        os.makedirs(origin_png_dir)

    for idx, (img, _) in enumerate(dataset):
        if isinstance(img, torch.Tensor): 
            img = denormalize(img)
            img = transforms.ToPILImage()(img)
        img_path = os.path.join(origin_png_dir, f"origin_{idx+1:04d}.png")
        img.save(img_path)
    
    print(f"{len(os.listdir(origin_png_dir))} images are done.")


def calculate_inception_score(images, model, splits=10):
    model.eval()
    preds = []

    for img in images:
        with torch.no_grad():
            pred = model(img)
            preds.append(softmax(pred, dim=1).cpu().numpy())
    
    preds = np.concatenate(preds, axis=0)  
    split_scores = []

    for k in range(splits):
        part = preds[k * (len(preds) // splits): (k + 1) * (len(preds) // splits)]
        py = np.mean(part, axis=0)  
        scores = [entropy(pyx, py) for pyx in part] 
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)

def evalutae(config, origin_png_dir, output_dir):
    fid_value = fid_score.calculate_fid_given_paths(
        [origin_png_dir, output_dir],
        batch_size=config["FID_size"],
        device=config['device'],
        dims=config["dims"]
    )

    IS_value = 0
    return fid_value

