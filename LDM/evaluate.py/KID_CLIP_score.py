#%%
""" KIDr """
import torch
import numpy as np
from scipy.stats import wasserstein_distance
from torch.nn.functional import adaptive_avg_pool2d
from torchvision.models import inception_v3

# Load Inception v3 model
model = inception_v3(pretrained=True, transform_input=False)
model.eval()

def extract_features(images, model, batch_size=32):
    features = []
    for i in range(0, len(images), batch_size):
        batch = torch.tensor(images[i:i+batch_size])
        with torch.no_grad():
            pred = model(batch)
        features.append(pred.cpu().numpy())
    return np.concatenate(features)

def calculate_kid(features1, features2):
    mean1, mean2 = np.mean(features1, axis=0), np.mean(features2, axis=0)
    cov1, cov2 = np.cov(features1, rowvar=False), np.cov(features2, rowvar=False)
    return wasserstein_distance(mean1, mean2)

# Example usage
# images1: Generated images (numpy array)
# images2: Real images (numpy array)
features1 = extract_features(images1, model)
features2 = extract_features(images2, model)
kid = calculate_kid(features1, features2)
print(f"KID: {kid}")

#%%
""" CLIP Score """
import torch
import clip
from PIL import Image

# Load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

def calculate_clip_score(image, text):
    image_input = preprocess(image).unsqueeze(0).to(device)
    text_input = clip.tokenize([text]).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image_input)
        text_features = model.encode_text(text_input)

    # Normalize features
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    # Calculate cosine similarity
    score = (image_features @ text_features.T).item()
    return score

# Example usage
image = Image.open("generated_image.jpg")  # Replace with your image
text = "A cat sitting on a chair"          # Replace with your caption
clip_score = calculate_clip_score(image, text)
print(f"CLIP Score: {clip_score}")
