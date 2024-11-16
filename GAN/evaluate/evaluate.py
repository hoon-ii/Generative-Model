#%%
import os
import torch
import torchvision.transforms as transforms
from pytorch_fid import fid_score
from modules.utils import denormalize
def convert2png(dataset, origin_png_dir):
    """
    데이터셋의 이미지를 PNG로 변환 및 저장.
    Args:
        config: dict, 설정 값 포함 (img_size 등).
        dataset: torchvision Dataset, 변환할 데이터셋.
        output_dir: str, PNG 이미지 저장 경로.
    """
    if not os.path.exists(origin_png_dir):
        os.makedirs(origin_png_dir)

    for idx, (img, _) in enumerate(dataset):
        if isinstance(img, torch.Tensor):  # Tensor인지 확인
            img = denormalize(img)
            img = transforms.ToPILImage()(img)
        img_path = os.path.join(origin_png_dir, f"origin_{idx+1:04d}.png")
        img.save(img_path)
    
    print(f"{len(os.listdir(origin_png_dir))}개의 MNIST 원본 이미지가 PNG로 저장되었습니다.")

def calculate_fid(config, origin_png_dir, output_dir):
    """
    FID 계산 함수
    real_images_dir: 실제 이미지 저장 경로
    generated_images_dir: 생성된 이미지 저장 경로
    """
    fid_value = fid_score.calculate_fid_given_paths(
        [origin_png_dir, output_dir],
        batch_size=config["FID_size"],
        device=config['device'],
        dims=config["dims"]
    )
    return fid_value
