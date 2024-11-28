import os
import random
import torch
from collections import deque
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np
import torch.nn.functional as F
import torchvision.transforms.functional as F2

CUSTOM_DATA_DIR = 'data/custom'
DEFOCUS_DIR = 'data/1000'
DATASET_DIR = 'data/'
BLUR_DIR = "data/dataset"
TEST_DIR = "data/dataset/test"

def pad_to_square(image):
    """이미지를 정사각형으로 패딩합니다."""
    c, h, w = image.shape
    max_dim = max(h, w)
    pad_h = max_dim - h
    pad_w = max_dim - w
    padding = [pad_w // 2, pad_h // 2, pad_w - pad_w // 2, pad_h - pad_h // 2]  # 좌우, 상하
    return F2.pad(image, padding, fill=0)


def load_images(mode='custom', image_size=(256, 256), max_images=None):
    """
    mode에 따라 custom_images/, dataset/, 또는 blur/ 폴더에서 이미지를 불러와서 Tensor 형태로 반환.
    
    Args:
        mode (str): 'custom', 'dataset' 또는 'blur' 중 선택하여 이미지 소스를 결정.
        image_size (tuple): 불러올 이미지의 크기. 기본값은 (256, 256).
        max_images (int): 불러올 최대 이미지 수. 기본값은 None(제한 없음).
    """
    if mode == 'custom':
        data_dir = CUSTOM_DATA_DIR
    elif mode == 'defocus':
        data_dir = DEFOCUS_DIR
    elif mode == 'dataset':
        data_dir = DATASET_DIR
    elif mode == 'blur':
        data_dir = BLUR_DIR
    elif mode == 'test':
        data_dir = TEST_DIR
    else:
        raise ValueError("Mode should be 'custom', 'dataset', or 'blur'.")

    print(f"Loading images from {data_dir}...")
    transform = transforms.Compose([
        transforms.ToTensor(),  # Tensor로 변환
        transforms.Lambda(pad_to_square),  # 패딩 적용
        transforms.Resize((256, 256)),
        transforms.Normalize((0.5,), (0.5,))  # 정규화
    ])

    images = []
    count = 0  # 로드한 이미지 수를 추적
    for file_name in os.listdir(data_dir):
        if file_name.endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(data_dir, file_name)
            image = Image.open(image_path).convert("L")
            transformed_image = transform(image)

            # blur 모드일 경우 이미지와 경로 반환
            if mode == 'blur' or mode == 'test':
                images.append((transformed_image, image_path))
            else:
                images.append(transformed_image)

            count += 1
            if max_images and count >= max_images:
                break  # 최대 이미지 수에 도달하면 종료
            print(f"Loaded: {file_name}")

    if not images:
        print(f"No images found in {data_dir}.")
    print(f"Total images loaded: {len(images)}")
    return images
    
def compute_loss(batch, model, target_model, gamma, optimizer, device):
    """
    DQN 모델의 손실을 계산하고 모델을 업데이트하는 함수.

    Args:
        batch (tuple): 리플레이 버퍼에서 샘플링된 배치 데이터 (state, prev_action, action, reward, next_state, next_prev_action, done).
        model (torch.nn.Module): 현재 Q 네트워크 모델.
        target_model (torch.nn.Module): 타겟 Q 네트워크 모델.
        gamma (float): 할인율.
        optimizer (torch.optim.Optimizer): 모델의 옵티마이저.
        device (torch.device): 사용할 장치 (CPU 또는 GPU).

    Returns:
        loss (torch.Tensor): 계산된 손실 값.
    """
    # 배치 데이터 분리 및 GPU로 데이터 이동
    state, prev_action, action, reward, next_state, next_prev_action, done = batch
    state = state.to(device)
    next_state = next_state.to(device)
    prev_action = prev_action.to(device)
    next_prev_action = next_prev_action.to(device)
    action = action.to(device)
    reward = reward.to(device)
    done = done.float().to(device)  # done 텐서를 float 타입으로 변환

    # 현재 상태에서 Q 값 계산
    q_values = model(state, prev_action)
    current_q = q_values.gather(1, action.unsqueeze(1)).squeeze(1)  # 선택된 액션의 Q 값

    # 다음 상태에서 Q 값 계산
    next_q_values = target_model(next_state, next_prev_action)
    max_next_q = next_q_values.max(1)[0]
    expected_q = reward + (1 - done) * gamma * max_next_q  # 타겟 Q 값 계산

    # 손실 계산
    loss = F.mse_loss(current_q, expected_q.detach())

    # 모델 업데이트
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss

def apply_blur(image, focal_length, gt_focal_length):
    blur_amount = abs(focal_length - gt_focal_length)
    image_pil = transforms.ToPILImage()(image)
    if blur_amount == 0:
        return image
    else:
        image_blurred = cv2.GaussianBlur(np.array(image_pil), (0, 0), blur_amount)
        return transforms.ToTensor()(image_blurred)

def get_blur(focal_length, gt_focal_length):
    blur_amount = abs(focal_length - gt_focal_length)
    return blur_amount

def calculate_reward(prev_focal_length, curr_focal_length, gt_focal_length, prev_action, action):
    prev_distance = abs(prev_focal_length - gt_focal_length)
    curr_distance = abs(curr_focal_length - gt_focal_length)

    # 이전 거리가 0일 때, 현재 거리의 변화에 따른 보상
    if prev_distance == 0:
        if curr_distance == 0:
            return 10  # 목표를 유지했을 때 양수 보상
        else:
            return -10  # 목표에서 벗어난 경우 음수 보상

    # 이전보다 개선된 경우 1, 그렇지 않으면 -1 보상
    if curr_distance < prev_distance:
        if curr_distance == 0:
            return 10
        else:
            return 2  # 개선된 경우
    else:
        return -2  # 이전보다 멀어졌거나 개선되지 않은 경우
    
# def calculate_reward(prev_focal_length, curr_focal_length, gt_focal_length, prev_action, action):
#     prev_distance = abs(prev_focal_length - gt_focal_length)
#     curr_distance = abs(curr_focal_length - gt_focal_length)

#     # 1. 목표 유지 보상
#     if prev_distance == 0 and curr_distance == 0:
#         return 5  # 유지했지만 추가 보상 제한

#     # 2. 목표에 가까워진 경우
#     if curr_distance < prev_distance:
#         if curr_distance == 0:
#             return 10
#         elif action != prev_action:  # 행동 변화로 개선된 경우
#             return 3
#         else:
#             return 2  # 단순 개선

#     # 3. 변화 없음
#     if curr_distance == prev_distance:
#         return 0  # 변화 없음

#     # 4. 목표에서 멀어진 경우
#     if curr_distance > prev_distance:
#         return -2

#     # 5. 비합리적 큰 행동 변화에 대한 페널티
#     if abs(action - prev_action) > 1:
#         return -5  # 큰 변화 페널티

#     return -10  # 기타 예상치 못한 상황



def save_checkpoint(model, optimizer, episode, save_path):
    """
    모델 체크포인트를 저장하는 함수.
    
    Args:
        model (torch.nn.Module): 저장할 모델.
        optimizer (torch.optim.Optimizer): 모델의 옵티마이저.
        episode (int): 현재 에피소드 번호.
        save_path (str): 체크포인트를 저장할 폴더 경로.
    """
    # 체크포인트 파일 이름에 episode 번호를 포함
    checkpoint_filepath = os.path.join(save_path, f"latest_checkpoint.pth")
    
    # 체크포인트 저장
    torch.save({
        'episode': episode,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, checkpoint_filepath)
    
    print(f"Checkpoint saved at episode {episode} in {checkpoint_filepath}.")

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, prev_action, action, reward, next_state, next_prev_action, done):
        self.buffer.append((state, prev_action, action, reward, next_state, next_prev_action, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, prev_action, action, reward, next_state, next_prev_action, done = zip(*batch)
        return torch.stack(state), torch.stack(prev_action), torch.tensor(action), torch.tensor(reward), \
               torch.stack(next_state), torch.stack(next_prev_action), torch.tensor(done)

    def __len__(self):
        return len(self.buffer)