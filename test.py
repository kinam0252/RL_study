import os
import torch
import torch.nn.functional as F
import random
import shutil
from torchvision import transforms
from model import ResNetDQN, ComplexDQN, BlurDQN
from utils import apply_blur, calculate_reward, BlurDataset, transform
from PIL import Image
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def process_images_in_list(images):
    processed_images = []
    for image in images:
        image = image.to("cpu")
        # NumPy로 변환 후 PIL 이미지로 변환
        image = (image.numpy() * 255).astype(np.uint8).transpose(1, 2, 0)
        image = Image.fromarray(image)
        # Transform 적용
        processed_image = transform(image)
        processed_image = processed_image.to(device)
        processed_images.append(processed_image)
    return processed_images

# 모델 체크포인트 로드
def load_checkpoint(checkpoint_path, model):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    episode = checkpoint['episode']
    print(f"Loaded checkpoint from episode {episode}.")
    return model

def initialize_from_pretrained(model, checkpoint_path, device):
    pretrained_weights = torch.load(checkpoint_path, map_location=device)["model_state_dict"]
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_weights.items() if k in model_dict}
    print(f"Pretrained keys: {pretrained_dict.keys()}")
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    model.freeze_shared_layers()
    print(f"Model initialized from pretrained weights at {checkpoint_path}")

def load_images_from_checkpoint(checkpoint_path, image_size=(256, 256)):
    images_dir = os.path.join(os.path.dirname(checkpoint_path), "images")
    if not os.path.exists(images_dir):
        raise FileNotFoundError(f"No 'images' directory found at {images_dir}")
    
    images = []
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
    ])
    
    for image_file in os.listdir(images_dir):
        image_path = os.path.join(images_dir, image_file)
        if image_file.endswith((".png", ".jpg", ".jpeg")):
            # Grayscale로 이미지를 로드
            image = Image.open(image_path).convert("L")  # "L" 모드는 Grayscale
            images.append(transform(image))
            print(f"Loaded Grayscale image: {image_path}")

    if not images:
        raise FileNotFoundError(f"No valid image files found in {images_dir}")
    
    return images


# 개별 트라이얼 폴더 생성 및 이미지 저장
def save_image(image, save_path,):
    to_pil = transforms.ToPILImage()
    image_pil = to_pil(image)
    image_resized = image_pil.resize((128, 128), resample=Image.BILINEAR)
    image_resized.save(save_path)

def create_trial_folder(sample_dir="sample", trial_name="trial"):
    trial_num = len(os.listdir(sample_dir)) + 1
    trial_folder = os.path.join(sample_dir, f"{trial_name}_{trial_num}")
    os.makedirs(trial_folder, exist_ok=True)
    return trial_folder

# 트라이얼 수행
def run_trial(model, trial_folder, image, gt_focal_length, num_steps, org_image):
    focal_0 = max(0, min(30, gt_focal_length + random.choice([-5, 5])))
    focal_1 = max(0, min(30, focal_0 + random.choice([-1, 0, 1])))
    image_0 = apply_blur(image, focal_0, gt_focal_length).to(device)
    image_1 = apply_blur(image, focal_1, gt_focal_length).to(device)
    
    image_0 = process_images_in_list([image_0])[0]
    image_1 = process_images_in_list([image_1])[0]
    
    prev_action = F.one_hot(torch.tensor(random.randint(0, 2)), num_classes=3).float().to(device)
    state = torch.cat([image_0.unsqueeze(0), image_1.unsqueeze(0)], dim=0).unsqueeze(0).to(device)
    total_reward = 0

    for step in range(1, num_steps + 1):
        with torch.no_grad():
            q_values = model(state, prev_action.unsqueeze(0), mode="Q")
            action = q_values.argmax().item()

        new_focal = max(0, min(30, focal_1 + (1 if action == 1 else -1 if action == 0 else 0))) 
        new_image = apply_blur(image, new_focal, gt_focal_length).to(device)
        new_image = process_images_in_list([new_image])[0]
        blurred_image = apply_blur(org_image, new_focal, gt_focal_length)
        
        reward = calculate_reward(focal_1, new_focal, gt_focal_length, prev_action.argmax().item(), action)
        print(f"Step {step}: Focal Length: {focal_1} -> {new_focal}, Action: {action} Reward: {reward}")
        total_reward += reward

        next_state = torch.cat([image_1.unsqueeze(0), new_image.unsqueeze(0)], dim=0).unsqueeze(0).to(device)
        next_prev_action = F.one_hot(torch.tensor(action), num_classes=3).float().to(device)

        save_image(blurred_image, os.path.join(trial_folder, f"step_{step}.png"))

        state, prev_action, image_1, focal_1 = next_state, next_prev_action, new_image, new_focal

    print(f"Trial completed. Total Reward: {total_reward}")

# DQN 테스트 함수
def test_dqn(checkpoint_path, num_trials=3, num_steps=10, num_samples=100):
    pretrained_checkpoint = "blur_pretrain/checkpoint/blur_predictor_epoch_250.pth"
    
    model_mode = "blur"
    if model_mode == "complex":
        model = ComplexDQN(input_image_channels=6, action_size=3).to(device)
    elif model_mode == "resnet":
        model = ResNetDQN(input_image_channels=1, action_size=3).to(device)
        # initialize_from_pretrained(model, pretrained_checkpoint, device)
    elif model_mode == "blur":
        model = BlurDQN(input_image_channels=3, action_size=3).to(device)

    model = load_checkpoint(checkpoint_path, model)

    # 학습 시 사용한 이미지 로드
    dataset_dir = "data/dataset_100"
    dataset = BlurDataset(dataset_dir, mode="Q")
    
    sample_dir = "sample"
    if os.path.exists(sample_dir):
        shutil.rmtree(sample_dir)
    os.makedirs(sample_dir)

    datasets = [dataset[i] for i in range(num_samples)]

    for image, org_image in datasets:
        org_image = transforms.ToTensor()(org_image)
        for trial in range(num_trials):
            trial_folder = create_trial_folder(sample_dir=sample_dir, trial_name="test")
            gt_focal_length = random.randint(5, 25)
            print(f"Starting trial {trial + 1} for with GT Focal Length: {gt_focal_length}")
            run_trial(model, trial_folder, image, gt_focal_length, num_steps, org_image)

if __name__ == "__main__":
    checkpoint_path = "/home/nas2_userG/junhahyung/kkn/RL_study/checkpoints/latest_checkpoint.pth"
    test_dqn(checkpoint_path, num_trials=5, num_steps=10)