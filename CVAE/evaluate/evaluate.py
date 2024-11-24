#%%
import os
import torch
import torchvision.transforms as transforms
from pytorch_fid import fid_score
from modules.utils import denormalize
# import inception_v3
from torch.nn.functional import softmax
import numpy as np

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

def fid_eval(config, origin_png_dir, output_dir):
    fid_value = fid_score.calculate_fid_given_paths(
        [origin_png_dir, output_dir],
        batch_size=config["FID_size"],
        device=config['device'],
        dims=config["dims"]
    )
    return fid_value


def inception_score(config, output_dir):
    inception_model = inception_v3(pretrained=True, transform_input=False).eval()
    inception_model = inception_model.to(config['device'])

    preprocess = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    preds = []
    with torch.no_grad():
        for img, _ in output_dir:
            if isinstance(img, torch.Tensor):
                img = preprocess(img).unsqueeze(0).to(config['device'])
            logits = inception_model(img)
            preds.append(softmax(logits, dim=1).cpu().numpy())
    
    preds = np.concatenate(preds, axis=0)

    p_y = np.mean(preds, axis=0)

    splits = config.get("splits", 10)
    scores = []
    split_size = len(preds) // splits
    for i in range(splits):
        part = preds[i * split_size:(i + 1) * split_size]
        kl = part * (np.log(part + 1e-8) - np.log(p_y + 1e-8))
        scores.append(np.exp(np.mean(np.sum(kl, axis=1))))
    
    return np.mean(scores), np.std(scores)