import torch
import torch.optim as optim
import torch.nn.functional as F
import random
from utils import load_images, apply_blur, calculate_reward, ReplayBuffer, save_checkpoint, compute_loss, BlurDataset
from model import ResNetDQN, ComplexDQN
from datetime import datetime, timezone, timedelta
import os
import numpy as np
from torchvision.utils import save_image
import shutil

DEBUG = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_checkpoint_folder(checkpoint_dir="checkpoints"):
    kst = datetime.now(timezone(timedelta(hours=9)))
    # timestamp = kst.strftime('%Y-%m-%d_%H-%M')
    # save_path = os.path.join(checkpoint_dir, timestamp)
    os.makedirs(checkpoint_dir, exist_ok=True)
    return checkpoint_dir

def initialize_from_pretrained(model, checkpoint_path, device):
    pretrained_weights = torch.load(checkpoint_path, map_location=device)
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_weights.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    print(f"Model initialized from pretrained weights at {checkpoint_path}")
    
def load_from_checkpoint(model, optimizer, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    model.freeze_shared_layers()
    print(f"Model loaded from checkpoint at {checkpoint_path}")

def save_images_to_folder(images, folder_path="data/custom"):
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
    os.makedirs(folder_path, exist_ok=True)

    for idx, image in enumerate(images):
        image_path = os.path.join(folder_path, f"image_{idx}.png")
        save_image(image, image_path)
        print(f"Saved {image_path}")

def train_dqn(num_episodes=100000, batch_size=100, replay_batch_size=256, gamma=0.99, epsilon_start=0.7, 
              epsilon_end=0.01, epsilon_decay=0.99, checkpoint_interval=100, mode='defocus',):
    dataset_dir = "data/dataset_100"
    dataset = BlurDataset(dataset_dir, mode="Q")
    def create_dataloader(batch_size):
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    dataloader = create_dataloader(batch_size)

    replay_buffer = ReplayBuffer(10000)
    pretrained_checkpoint = "blur_pretrain/checkpoints/best_model_checkpoint.pth"
    
    model_mode = "complex"
    
    if model_mode == "resnet":  
        model = ResNetDQN(input_image_channels=6, action_size=3).to(device)
        target_model = ResNetDQN(input_image_channels=1, action_size=3).to(device)
        initialize_from_pretrained(model, pretrained_checkpoint, device)
        initialize_from_pretrained(target_model, pretrained_checkpoint, device)
        target_model.load_state_dict(model.state_dict())
    elif model_mode == "complex":
        model = ComplexDQN(input_image_channels=6, action_size=3).to(device)
        target_model = ComplexDQN(input_image_channels=6, action_size=3).to(device)
        load_from_checkpoint(model, None, pretrained_checkpoint, device)
        model.freeze_shared_layers()
        target_model.load_state_dict(model.state_dict())
        
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    epsilon = epsilon_start

    checkpoint_folder = get_checkpoint_folder()
    
    # save_images_to_folder(dataset, folder_path=f"{checkpoint_folder}/images")
    
    for episode in range(num_episodes):
        batch_indices = random.sample(range(len(dataset)), batch_size)
        gt_focal_lengths = [random.randint(5, 25) for _ in range(batch_size)]
        images = [dataset[i] for i in batch_indices]
        # images = [images[0] for _ in range(batch_size)]

        focal_lengths = [max(0, min(30, gt_focal_lengths[i] + random.randint(-5, 5))) for i in range(batch_size)]
        next_focal_lengths = [max(0, min(30, focal_lengths[i] + random.choice([-1, 1, 0]))) for i in range(batch_size)]

        gt_maintain_counts = [0 for _ in range(batch_size)]

        image_0s = [apply_blur(images[i], focal_lengths[i], gt_focal_lengths[i]).to(device) for i in range(batch_size)]
        image_1s = [apply_blur(images[i], next_focal_lengths[i], gt_focal_lengths[i]).to(device) for i in range(batch_size)]

        # image_0s = [img if img.dim() == 3 else img.unsqueeze(0) for img in image_0s]
        # image_1s = [img if img.dim() == 3 else img.unsqueeze(0) for img in image_1s]

        # print(f"image_0s: {image_0s[0].shape}, image_1s: {image_1s[0].shape}")

        prev_actions = [F.one_hot(torch.tensor(random.randint(0, 2)), num_classes=3).float().to(device) for _ in range(batch_size)]
        states = [torch.cat([image_0s[i], image_1s[i]], dim=0) for i in range(batch_size)]
        states = torch.stack(states)  # Shape: [Batch, 2, 256, 256]
        # print(f"states: {states.shape}")
        prev_actions = torch.stack(prev_actions)  # Shape: [Batch, 3]
        total_rewards = [0 for _ in range(batch_size)]
        states = states.to(device)
        prev_actions = prev_actions.to(device)
        
        for step in range(10):
            actions = []
            if random.random() < epsilon:
                actions = [random.randint(0, 2) for _ in range(batch_size)]
                if DEBUG:
                    print(f"[DEBUG] Random actions taken. actions: {actions}")
            else:
                q_values = model(states, prev_actions)  # Shape: [Batch, Action_Size]
                actions = q_values.argmax(dim=1).tolist()
                if DEBUG:
                    print(f"[DEBUG] Model actions taken. actions: {actions}")

            new_focal_lengths = [
                max(0, min(30, next_focal_lengths[i] + (1 if actions[i] == 1 else -1 if actions[i] == 0 else 0)))
                for i in range(batch_size)
            ]
            new_images = [
                apply_blur(images[i], new_focal_lengths[i], gt_focal_lengths[i]).to(device)
                for i in range(batch_size)
            ]
            new_images = [img if img.dim() == 3 else img.unsqueeze(0) for img in new_images]

            rewards = [
                calculate_reward(next_focal_lengths[i], new_focal_lengths[i], gt_focal_lengths[i],
                                 prev_actions[i].argmax().item(), actions[i])
                for i in range(batch_size)
            ]
            if DEBUG:
                print(f"[DEBUG] rewards: {rewards}")
            total_rewards = [total_rewards[i] + rewards[i] for i in range(batch_size)]

            next_states = [torch.cat([image_1s[i], new_images[i]], dim=0) for i in range(batch_size)]
            next_states = torch.stack(next_states)  # Shape: [Batch, 2, 256, 256]
            next_prev_actions = [F.one_hot(torch.tensor(actions[i]), num_classes=3).float().to(device) for i in range(batch_size)]
            next_prev_actions = torch.stack(next_prev_actions)  # Shape: [Batch, 3]

            for i in range(batch_size):
                if new_focal_lengths[i] == gt_focal_lengths[i]:
                    gt_maintain_counts[i] += 1
                else:
                    gt_maintain_counts[i] = 0

                replay_buffer.push(states[i].to("cpu"), prev_actions[i], actions[i], rewards[i], next_states[i].to("cpu"), next_prev_actions[i], False)

            if len(replay_buffer) >= replay_batch_size:
                # print(f"Training model with replay buffer of size {len(replay_buffer)}")
                batch = replay_buffer.sample(replay_batch_size)
                loss = compute_loss(batch, model, target_model, gamma, optimizer, device)

            if all(count >= 3 for count in gt_maintain_counts):
                break

            states, prev_actions, image_1s, next_focal_lengths = next_states, next_prev_actions, new_images, new_focal_lengths

        print(f"Episode {episode + 1}/{num_episodes} - Average Total Reward: {sum(total_rewards) / batch_size}")

        if episode % 10 == 0:
            target_model.load_state_dict(model.state_dict())

        if episode != 0 and episode % checkpoint_interval == 0:
            save_checkpoint(model, optimizer, episode, checkpoint_folder)

        epsilon = max(epsilon_end, epsilon * epsilon_decay)

    save_checkpoint(model, optimizer, episode, checkpoint_folder)
    print("Training completed.")

train_dqn()
