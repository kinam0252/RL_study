import os
import torch
import torch.nn.functional as F
import random
import shutil
from torchvision import transforms
from model import ResNetDQN, ComplexDQN, RLModel
from utils import apply_blur, calculate_reward
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 모델 체크포인트 로드
def load_checkpoint(checkpoint_path, model, optimizer):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    episode = checkpoint['episode']
    print(f"Loaded checkpoint from episode {episode}.")
    return model, optimizer, episode

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
def save_image(image, save_path):
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
def run_trial(model, trial_folder, image, gt_focal_length, num_steps):
    focal_0 = max(0, min(30, gt_focal_length + random.choice([-5, 5])))
    focal_1 = max(0, min(30, focal_0 + random.choice([-1, 0, 1])))
    focal_0 = list([focal_0])
    focal_1 = list([focal_1])
    gt_focal_length = list([gt_focal_length])
    blur_0 = torch.abs(torch.tensor(focal_0) - torch.tensor(gt_focal_length)).unsqueeze(0)
    blur_1 = torch.abs(torch.tensor(focal_1) - torch.tensor(gt_focal_length)).unsqueeze(0)
    prev_action = F.one_hot(torch.tensor(random.randint(0, 2)), num_classes=3).float().to(device)
    state = torch.cat([blur_0, blur_1], dim=1).to(device)
    total_reward = 0

    for step in range(1, num_steps + 1):
        with torch.no_grad():
            q_values = model(state, prev_action.unsqueeze(0))
            action = q_values.argmax().item()

        new_focal = max(0, min(30, focal_1[0] + (1 if action == 1 else -1 if action == 0 else 0))) 
        # new_image = apply_blur(image, new_focal, gt_focal_length).to(device)
        new_focal = list([new_focal])
        new_blur = torch.abs(torch.tensor(new_focal) - torch.tensor(gt_focal_length)).unsqueeze(0)
        
        # reward = calculate_reward(focal_1, new_focal, gt_focal_length, prev_action.argmax().item(), action)
        print(f"Step {step}: Focal Length: {focal_1[0]} -> {new_focal[0]}, Action: {action}")
        # total_reward += reward

        # next_state = torch.cat([image_1, new_image], dim=0).unsqueeze(0).to(device)
        next_state = torch.cat([blur_1, new_blur], dim=1).to(device)
        next_prev_action = F.one_hot(torch.tensor(action), num_classes=3).float().to(device)

        # save_image(new_image, os.path.join(trial_folder, f"step_{step}.png"))

        state, prev_action, blur_1, focal_1 = next_state, next_prev_action, new_blur, new_focal

    print(f"Trial completed. Total Reward: {total_reward}")

# DQN 테스트 함수
def test_dqn(checkpoint_path, num_trials=3, num_steps=10):
    pretrained_checkpoint = "blur_pretrain/checkpoint/blur_predictor_epoch_250.pth"
    
    model_mode = "simple"
    if model_mode == "complex":
        model = ComplexDQN(input_image_channels=2, action_size=3).to(device)
    elif model_mode == "resnet":
        model = ResNetDQN(input_image_channels=1, action_size=3).to(device)
        # initialize_from_pretrained(model, pretrained_checkpoint, device)
    elif model_mode == "simple":
        model = RLModel().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model, optimizer, _ = load_checkpoint(checkpoint_path, model, optimizer)

    # # 학습 시 사용한 이미지 로드
    # images = load_images_from_checkpoint(checkpoint_path, image_size=(256, 256))
    # sample_dir = "sample"
    # if os.path.exists(sample_dir):
    #     shutil.rmtree(sample_dir)
    # os.makedirs(sample_dir)
    image = None
    trial_folder = None
    for trial in range(num_trials):
        # trial_folder = create_trial_folder(sample_dir=sample_dir, trial_name="test")
        gt_focal_length = random.randint(5, 25)
        print(f"Starting trial {trial + 1} for with GT Focal Length: {gt_focal_length}")
        run_trial(model, trial_folder, image, gt_focal_length, num_steps)

if __name__ == "__main__":
    checkpoint_path = "/home/nas2_userG/junhahyung/kkn/RL_study/laboratory/checkpoints/2024-11-29_02-12/latest_checkpoint.pth"
    test_dqn(checkpoint_path, num_trials=10, num_steps=10)