import os
import torch
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
from model import ComplexDQN  # 학습된 모델 로드

# 데이터셋 클래스 정의 (기존 코드와 동일)
class BlurDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_dir, transform=None):
        self.dataset_dir = dataset_dir
        self.image_files = [f for f in os.listdir(dataset_dir) if f.endswith('.jpg')]
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # 이미지와 레이블 파일 이름
        image_path = os.path.join(self.dataset_dir, self.image_files[idx])
        label_path = os.path.splitext(image_path)[0] + ".txt"

        # 이미지 읽기
        image = Image.open(image_path).convert("L")  # Grayscale 전환

        # 이미지를 두 개로 분리
        w, h = image.size
        half_w = w // 2
        image1 = image.crop((0, 0, half_w, h))  # 왼쪽 이미지
        image2 = image.crop((half_w, 0, w, h))  # 오른쪽 이미지

        # Transform 각각 적용
        if self.transform:
            image1 = self.transform(image1)
            image2 = self.transform(image2)

        # 이미지 스택
        images = torch.stack([image1, image2], dim=0)

        # 레이블 읽기
        with open(label_path, 'r') as f:
            labels = f.readline().strip().split()
            blur_label1, blur_label2 = map(float, labels)

        return images, torch.tensor([blur_label1, blur_label2], dtype=torch.float32), image_path

# 설정
dataset_dir = "data/dataset"
samples_dir = "blur_pretrain/samples/"
os.makedirs(samples_dir, exist_ok=True)
model_path = "blur_pretrain/checkpoints/best_model_checkpoint.pth"  # 학습된 모델 경로
num_samples = 10  # 추출할 이미지 개수

# 데이터 전처리
transform = transforms.Compose([
    transforms.ToTensor(),  # Tensor로 변환
    transforms.Resize((256, 256)),
    transforms.Normalize((0.5,), (0.5,))  # 정규화
])

# 데이터셋 및 데이터 로더
dataset = BlurDataset(dataset_dir, transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

# 모델 로드
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ComplexDQN(input_image_channels=2, action_size=3)  # Grayscale이므로 채널 수 1
checkpoint = torch.load(model_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

# 샘플 추출 및 저장
font_path = "/scratch/slurm-user25-kaist/user/kinamkim/Roboto-Bold.ttf"  # 시스템 폰트 경로 (수정 필요)
for idx, (images, gt_blurs, image_path) in enumerate(dataloader):
    if idx >= num_samples:
        break

    # 이미지 이름 추출
    image_name = os.path.basename(image_path[0])

    # 모델 예측
    images = images.view(images.size(0), -1, images.size(3), images.size(4)).to(device)  # (batch_size, C * 2, H, W)
    with torch.no_grad():
        pred_blurs = model(images, mode="blur").cpu().numpy()

    # 원본 이미지 로드
    original_image = Image.open(image_path[0]).convert("RGB")

    # 시각화 생성
    combined_image = Image.new("RGB", (original_image.width, original_image.height + 100), (255, 255, 255))
    combined_image.paste(original_image, (0, 0))

    # 텍스트 추가
    draw = ImageDraw.Draw(combined_image)
    font = ImageFont.truetype(font_path, size=20)
    text = (
        f"GT Blur: {gt_blurs[0, 0]:.2f}, {gt_blurs[0, 1]:.2f}\n"
        f"Pred Blur: {pred_blurs[0, 0]:.2f}, {pred_blurs[0, 1]:.2f}"
    )
    draw.text((10, original_image.height + 10), text, fill=(0, 0, 0), font=font)

    # 저장
    combined_image.save(os.path.join(samples_dir, f"sample_{idx + 1}_{image_name}"))

print(f"{num_samples} samples saved to {samples_dir}")
