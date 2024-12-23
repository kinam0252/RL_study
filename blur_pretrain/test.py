import os
import torch
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
from model import BlurDQN  # 모델 불러오기
from utils import BlurDataset
from torchvision.transforms import Normalize

def initialize_from_pretrained(model, checkpoint_path, device):
    pretrained_weights = torch.load(checkpoint_path, map_location=device)["model_state_dict"]
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_weights.items() if k in model_dict}
    print(f"Pretrained keys: {pretrained_dict.keys()}")
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    model.freeze_shared_layers()
    print(f"Model initialized from pretrained weights at {checkpoint_path}")

class ReverseNormalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)  # 정규화를 해제
        return tensor

# 결과 저장 경로 생성
sample_dir = "blur_pretrain/samples/"
os.makedirs(sample_dir, exist_ok=True)

# 체크포인트 로드
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BlurDQN(input_image_channels=3, action_size=3)
model.to(device)

checkpoint_path = "blur_pretrain/checkpoints/best_model_checkpoint.pth"
if os.path.exists(checkpoint_path):
    initialize_from_pretrained(model, checkpoint_path, device)
    print(f"Checkpoint loaded from {checkpoint_path}")
else:
    raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

# 데이터셋 경로 및 전처리
dataset_dir = "data/dataset_100"
dataset = BlurDataset(dataset_dir)

# 시각화 함수
def visualize_and_save(image, pred_blur, gt_blur, index, save_dir):
    """이미지와 예측값, GT값을 하나의 이미지로 시각화하여 저장"""
    reverse_normalize = ReverseNormalize(mean=(0.5,), std=(0.5,))
    
    # 이미지 해제
    image_tensor = reverse_normalize(image[0:3])  # 첫 번째 이미지만 사용

    # 이미지 데이터를 [0, 255] 범위로 변환 후 PIL 이미지로 변환
    image = transforms.ToPILImage()(image_tensor.clamp(0, 1))

    # 텍스트 영역 추가
    text_box_height = 50  # 텍스트 박스 높이
    total_height = image.height + text_box_height
    final_image = Image.new("RGB", (image.width, total_height), color="black")
    final_image.paste(image, (0, 0))

    # Blur 값을 텍스트 박스에 추가
    draw = ImageDraw.Draw(final_image)
    font = ImageFont.load_default()
    text = f"Predicted Blur: {round(pred_blur.item())}\nGT Blur: {gt_blur}"
    text_x = 10
    text_y = image.height + 10  # 텍스트 박스 내부 여백
    draw.text((text_x, text_y), text, fill="white", font=font)

    # 저장
    save_path = os.path.join(save_dir, f"sample_{index + 1}.png")
    final_image.save(save_path)
    print(f"Saved sample {index + 1} at {save_path}")


# 테스트
model.eval()
num_samples = 100  # 사용자 지정 샘플 개수
with torch.no_grad():
    for i in range(num_samples):
        images, gt_blur = dataset[i]  
        images = images.view(1, 1, 3, images.size(1), images.size(2)).to(device)

        pred_blur = model(images, mode="blur").squeeze(0)  # 예측 결과
        visualize_and_save(images[0][0], pred_blur, gt_blur, i, sample_dir)
    
print(f"All samples saved in {sample_dir}")
