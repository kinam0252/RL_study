import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from PIL import Image
from model import ComplexDQN  # 모델 불러오기

import torchvision.transforms.functional as F
from utils import BlurDataset

def pad_to_square(image):
    """이미지를 정사각형으로 패딩합니다."""
    c, h, w = image.shape
    max_dim = max(h, w)
    pad_h = max_dim - h
    pad_w = max_dim - w
    padding = [pad_w // 2, pad_h // 2, pad_w - pad_w // 2, pad_h - pad_h // 2]  # 좌우, 상하
    return F.pad(image, padding, fill=0)

# 하이퍼파라미터 및 경로 설정
dataset_dir = "data/dataset_100"
initial_batch_size = 64
min_batch_size = 16
learning_rate = 1e-4
num_epochs = 1000
save_interval = 10  # 몇 에포크마다 체크포인트 저장할지 설정

# 데이터 전처리
transform = transforms.Compose([
    transforms.ToTensor(),  # Tensor로 변환
    transforms.Lambda(pad_to_square),  # 패딩 적용
    transforms.Resize((256, 256)),
    transforms.Normalize((0.5,), (0.5,))  # 정규화
])

# 데이터셋 및 데이터 로더 생성 함수
dataset = BlurDataset(dataset_dir)
def create_dataloader(batch_size):
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 모델 초기화
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ComplexDQN(input_image_channels=6, action_size=3)  # Grayscale이므로 채널 수 1
model.to(device)
print(f"default device: {device}")

# 손실 함수 및 옵티마이저 초기화
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 학습률 스케줄러 초기화
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)

checkpoint_dir = "blur_pretrain/checkpoints/"
os.makedirs(checkpoint_dir, exist_ok=True)

# 체크포인트 로드 함수
def load_checkpoint(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    print(f"Checkpoint loaded: Starting from epoch {start_epoch}")
    return start_epoch

# 체크포인트 경로 확인
latest_checkpoint_path = f"{checkpoint_dir}/latest_checkpoint.pth"
best_checkpoint_path = f"{checkpoint_dir}/best_model_checkpoint.pth"
start_epoch = 0
best_loss = float('inf')  # 초기값으로 무한대 설정

if os.path.exists(best_checkpoint_path):
    start_epoch = load_checkpoint(best_checkpoint_path)

# 학습 루프
current_batch_size = initial_batch_size
dataloader = create_dataloader(current_batch_size)

warmup = 50
batch_warmup = 50

for epoch in range(start_epoch, num_epochs):
    model.train()
    running_loss = 0.0

    for images, labels in dataloader:
        images = images.view(images.size(0), -1, images.size(3), images.size(4))  # (batch_size, C * 2, H, W)
        images, labels = images.to(device), labels.to(device)

        # 예측
        outputs = model(images, mode="blur")

        # 손실 계산
        loss = criterion(outputs, labels)
        running_loss += loss.item()

        # 역전파 및 가중치 업데이트
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # 에폭별 평균 손실 계산
    avg_loss = running_loss / len(dataloader)
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")

    # 배치 크기 조정
    if epoch > warmup and (epoch + 1) % batch_warmup == 0 and current_batch_size > min_batch_size:
        current_batch_size = max(min_batch_size, current_batch_size // 2)
        dataloader = create_dataloader(current_batch_size)
        print(f"Batch size reduced to {current_batch_size}")
        # latest checkpoint 저장
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }, latest_checkpoint_path)
        print(f"Latest checkpoint saved at {latest_checkpoint_path}")

    # 스케줄러 업데이트
    scheduler.step(avg_loss)

    # best checkpoint 업데이트
    if epoch > warmup and avg_loss < best_loss:
        print(f"Saving best checkpoint at {best_checkpoint_path}")
        best_loss = avg_loss
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }, best_checkpoint_path)
        print(f"Best checkpoint updated at {best_checkpoint_path} with loss {best_loss:.4f}")

# 최종 모델 저장
torch.save(model.state_dict(), f"{checkpoint_dir}/blur_final.pth")
print("모델 학습 완료 및 최종 저장됨!")
