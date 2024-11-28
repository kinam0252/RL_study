import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import StepLR
import numpy as np

# 1. Hyperparameters and settings
image_batch_size = 32  # 예시 배치 사이즈
learning_rate = 1e-3
num_epochs = 100000
checkpoint_dir = 'laboratory/checkpoint'
os.makedirs(checkpoint_dir, exist_ok=True)

DEBUG = True

# 2. RL 환경: focal length와 blur amount로 학습
class FocalLengthEnv:
    def __init__(self, image_batch_size):
        self.image_batch_size = image_batch_size
        self.gt_focal_lengths = [random.randint(5, 25) for _ in range(image_batch_size)]  # 실제 focal length

    def get_blur_amount(self, focal_length, gt_focal_length):
        return abs(focal_length - gt_focal_length)  # blur amount = focal length 차이

    def reset(self):
        self.prev_focal_lengths = [max(0, min(30, self.gt_focal_lengths[i] + random.randint(-5, 5))) for i in range(self.image_batch_size)]
        self.prev_actions = [random.choice([0, 1, -1]) for _ in range(self.image_batch_size)]
        self.curr_focal_lengths = [max(0, min(30, self.prev_focal_lengths[i] + self.prev_actions[i])) for i in range(self.image_batch_size)]
        return self.prev_focal_lengths, self.prev_actions, self.curr_focal_lengths

    def step(self, actions):
        # 새로운 focal length를 actions에 따라 업데이트
        self.next_focal_lengths = [
            max(0, min(30, self.curr_focal_lengths[i] + (1 if actions[i] == 1 else -1 if actions[i] == 0 else 0)))
            for i in range(self.image_batch_size)
        ]
        
        # 현재 blur 값 계산
        next_blur = [self.get_blur_amount(self.next_focal_lengths[i], self.gt_focal_lengths[i]) for i in range(self.image_batch_size)]

        # 보상 계산
        rewards = []
        for i in range(self.image_batch_size):
            # 현재 focal length가 gt focal length와 일치하면 다음 focal length가 그대로 유지되는 경우에 보상
            if self.next_focal_lengths[i] == self.gt_focal_lengths[i]:
                reward = 1  # 이미 gt focal length에 맞춰졌으면 양수 보상
            else:
                # 현재 focal length와 gt focal length 간의 차이
                current_diff = abs(self.curr_focal_lengths[i] - self.gt_focal_lengths[i])
                
                # 다음 focal length로 변경된 후 차이를 계산
                next_diff = abs(self.next_focal_lengths[i] - self.gt_focal_lengths[i])

                # 만약 next focal length가 더 가까워졌다면 보상 +1, 멀어졌다면 보상 -1
                if next_diff < current_diff:
                    reward = 1
                elif next_diff > current_diff:
                    reward = -1
                else:
                    reward = -1  # 차이가 그대로라면 0 보상

            rewards.append(reward)
        self.prev_focal_lengths = self.curr_focal_lengths
        self.curr_focal_lengths = self.next_focal_lengths
        self.prev_actions = actions
        return self.next_focal_lengths, rewards
    
    def get_prev_blur(self):
        return [self.get_blur_amount(self.prev_focal_lengths[i], self.gt_focal_lengths[i]) for i in range(self.image_batch_size)]
    
    def get_curr_blur(self):
        return [self.get_blur_amount(self.curr_focal_lengths[i], self.gt_focal_lengths[i]) for i in range(self.image_batch_size)]

# 3. RL 모델: 이전 blur, 현재 blur, action을 받아서 새로운 action을 예측
class RLModel(nn.Module):
    def __init__(self):
        super(RLModel, self).__init__()
        
        # 각 입력을 고차원으로 변환
        self.fc_input = nn.Linear(2, 64)
        
        # 결합된 차원을 처리하기 위한 중간 레이어들 (fc 레이어 개수를 절반으로 줄임)
        self.fc1 = nn.Linear(64, 64)  # 192 -> 256 (64*3 차원으로 수정)
        self.fc2 = nn.Linear(64, 64)  # 256 -> 512
        
        # 최종 출력: 3개의 action logits (각각 -1, 0, 1)
        self.fc_out = nn.Linear(64, 3)
    
    def forward(self, x):
        x1 = x[:, 0]
        x2 = x[:, 1]
        x3 = x[:, 2]
        x_diff = x2 - x1  # 이전 blur와 현재 blur의 차이
        x_diff = x_diff.unsqueeze(1)  # 차원 추가
        x3 = x3.unsqueeze(1)  # 차원 추가
        if DEBUG:
            print(f"[DEBUG] x_diff: {x_diff[0]} x3: {x3[0]}")
        x = torch.cat([x_diff, x3], dim=-1)  # 이전 blur, 현재 blur, action, blur 차이를 결합
        x = torch.relu(self.fc_input(x))

        # 결합된 입력을 중간 레이어에 통과시킴
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        
        # 최종 출력: 3개의 action logits
        x = self.fc_out(x)
        
        return x

# 4. Training function
def train(model, env, optimizer, scheduler, epsilon_start=0.7, epsilon_end=0.01, epsilon_decay=0.9, debug_mode=False):
    model.train()
    epoch_rewards = []
    epsilon = epsilon_start  # 초기 epsilon 값 설정

    for epoch in range(num_epochs):
        # 각 에폭마다 환경 초기화 (focal_lengths, gt_focal_lengths 등 초기화)
        prev_focal_lengths, prev_actions, curr_focal_lengths = env.reset()
        total_reward = 0

        for step in range(10):  # 각 에폭에서 10번의 step을 진행
            # prev_blur, curr_blur, actions를 결합하여 input_data 생성
            prev_blur = env.get_prev_blur()
            curr_blur = env.get_curr_blur()
            input_data = np.array([[prev_blur[i], curr_blur[i], prev_actions[i]] for i in range(image_batch_size)], dtype=np.float32)

            # torch tensor로 변환
            input_data = torch.tensor(input_data, dtype=torch.float32)

            # 랜덤 액션 또는 모델의 액션 선택
            if random.random() < epsilon:
                # 랜덤 액션 선택
                actions = [random.randint(0, 2) for _ in range(image_batch_size)]
                chosen_actions = actions  # 랜덤 선택한 actions을 chosen_actions에 할당
                if debug_mode:
                    print(f"[DEBUG] Random actions taken. actions: {actions[0]}")
            else:
                # 모델을 통한 액션 선택
                logits = model(input_data)
                prob_actions = torch.softmax(logits, dim=-1)
                chosen_actions = torch.argmax(prob_actions, dim=-1).numpy()  # 모델이 선택한 액션
                if debug_mode:
                    print(f"[DEBUG] Model actions taken. actions: {chosen_actions[0]}")

            # 환경에서 현재 focal length와 보상 계산
            _, rewards = env.step(chosen_actions)
            total_reward += sum(rewards)

            # Loss 계산 (보상 반영)
            loss = -torch.mean(torch.tensor(rewards, dtype=torch.float32, requires_grad=True))

            # Backpropagation
            optimizer.zero_grad()  # 기울기 초기화
            loss.backward()  # Backpropagation으로 기울기 계산
            optimizer.step()  # 파라미터 업데이트

            # Scheduler step
            scheduler.step()

            # 디버그 모드에서 첫 번째 배치 요소에 대한 자세한 정보 출력
            if DEBUG:
                # 첫 번째 배치 요소에 대해서만 출력
                i = 0  # 첫 번째 배치 요소 인덱스
                # action에 해당하는 명칭을 선택
                action_names = {0: "backward", 2: "stay", 1: "forward"}
                action_name = action_names[chosen_actions[i]]  # 액션의 숫자에 해당하는 명칭 선택

                print(f"[DEBUG] Epoch {epoch+1}/{num_epochs}, Step {step+1}/10")
                print(f"[DEBUG] focal_length (before): {env.prev_focal_lengths[i]}")
                print(f"[DEBUG] action: {action_name} (chosen action: {chosen_actions[i]})")
                print(f"[DEBUG] focal_length (after): {env.curr_focal_lengths[i]}")
                print(f"[DEBUG] GT focal_length: {env.gt_focal_lengths[i]}")
                print(f"[DEBUG] reward: {rewards[i]}")
                print(f"[DEBUG] loss: {loss.item()}")
                print("-----------------------------------------------------")

        epoch_rewards.append(total_reward)
        print(f'Epoch {epoch+1}/{num_epochs}, Reward: {total_reward}')

        # epsilon 값 감소 (점진적으로 epsilon 값을 줄임)
        epsilon = max(epsilon_end, epsilon * epsilon_decay)

        # 일정 에폭마다 체크포인트 저장
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'checkpoint.pth'))
            print(f"Checkpoint saved at epoch {epoch+1}")

        if DEBUG:
            assert False, "Stop here"  # 디버그 모드에서 멈춤


# 5. Optimizer 및 Scheduler 설정
model = RLModel()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = StepLR(optimizer, step_size=30, gamma=0.5)

# 6. Checkpoint 로드
checkpoint_path = os.path.join(checkpoint_dir, 'checkpoint.pth')
if os.path.exists(checkpoint_path):
    model.load_state_dict(torch.load(checkpoint_path))
    print("Loaded checkpoint from previous training")

# 7. RL 환경과 학습 진행
env = FocalLengthEnv(image_batch_size)
train(model, env, optimizer, scheduler)
