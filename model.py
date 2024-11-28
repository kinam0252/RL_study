import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18

DEBUG = False

class ResNetDQN(nn.Module):
    def __init__(self, input_image_channels, action_size, predict_mode="Q"):
        super(ResNetDQN, self).__init__()
        self.predict_mode = predict_mode

        # ResNet18 모델 초기화
        self.resnet = resnet18(pretrained=True)
        
        # 첫 레이어를 수정하여 2채널 이미지를 입력받도록 설정
        self.resnet.conv1 = nn.Conv2d(input_image_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        num_resnet_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()  # 마지막 FC 레이어 제거하여 특징 추출 부분만 사용

        # 공통으로 사용할 축소 레이어
        self.fc_shared = nn.Linear(num_resnet_features, 64)

        # Q-러닝 네트워크용 레이어
        self.leaky_relu = nn.LeakyReLU(0.1)
        self.fc_feature_to_64 = nn.Linear(64, 64)  # features를 64차원으로 변환
        self.fc_action_embedding = nn.Linear(action_size, 64)  # action을 64차원으로 embedding
        self.fc1 = nn.Linear(64 * 3, 128)  # 변환된 feature와 action embedding 결합
        self.fc2 = nn.Linear(128, action_size)

    def freeze_non_q_layers(self):
        """
        Q-러닝 네트워크 레이어를 제외한 나머지 파라미터 동결
        """
        for name, param in self.named_parameters():
            # Q-러닝 네트워크에 포함된 레이어는 requires_grad=True
            if any(q_layer in name for q_layer in ["fc_feature_to_64", "fc_action_embedding", "fc1", "fc2"]):
                param.requires_grad = True
            else:
                param.requires_grad = False

    def get_blur_mode_layers(self):
        """
        Blur 모드와 관련된 레이어 이름 가져오기
        """
        return [name for name, _ in self.named_parameters() if "resnet" in name or "fc_shared" in name or "fc_blur" in name]

    def forward(self, image_stack, previous_action=None):
        if self.predict_mode == "blur":
            # Blur 정도를 예측
            x = self.resnet(image_stack)
            return F.relu(self.fc_shared(x))  # 차원을 64로 축소하여 반환

        elif self.predict_mode == "Q":
            # Q 값 예측
            # image_stack은 [batch, channels*2, height, width] 형태
            # 이미지 두 개를 분리하여 배치처럼 처리
            img1, img2 = torch.chunk(image_stack, 2, dim=1)  # [batch, channels, height, width]로 분리

            # ResNet을 통해 특징 추출 후 축소 레이어 통과
            features1 = F.relu(self.fc_shared(self.resnet(img1)))  # [batch, 64]
            features2 = F.relu(self.fc_shared(self.resnet(img2)))  # [batch, 64]

            # 새로운 FC 레이어를 통해 features를 64차원으로 변환
            feature1_64 = F.relu(self.fc_feature_to_64(features1))  # [batch, 64]
            feature2_64 = F.relu(self.fc_feature_to_64(features2))  # [batch, 64]

            # Action을 64차원으로 embedding
            action_embed = F.relu(self.fc_action_embedding(previous_action))  # [batch, 64]

            # 변환된 feature와 action embedding을 결합
            combined_features = torch.cat([feature1_64, feature2_64, action_embed], dim=1)  # [batch, 64 * 3]
            
            # Fully connected layers로 처리
            x = F.relu(self.fc1(combined_features))  # [batch, 128]
            return self.fc2(x)  # [batch, action_size]



import torch
import torch.nn as nn
import torch.nn.functional as F

# class ComplexDQN(nn.Module):
#     def __init__(self, input_image_channels, action_size):
#         super(ComplexDQN, self).__init__()
#         # Deeper and more complex convolutional layers
#         self.conv1 = nn.Conv2d(input_image_channels, 64, kernel_size=5, stride=2, padding=2)
#         self.bn1 = nn.BatchNorm2d(64)
#         self.conv2 = nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2)
#         self.bn2 = nn.BatchNorm2d(128)
#         self.conv3 = nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2)
#         self.bn3 = nn.BatchNorm2d(256)
#         self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
#         self.bn4 = nn.BatchNorm2d(512)
        
#         # Fully connected layers for Q-value prediction
#         self.fc1 = nn.Linear(512 * 16 * 16, 1024)
#         self.action_embedding = nn.Linear(3, 64)
#         self.fc2 = nn.Linear(1024 + 64, 512)
#         self.fc3 = nn.Linear(512, action_size)

#         # Fully connected layers for blur value prediction
#         self.blur_fc1 = nn.Linear(512 * 16 * 16, 512)
#         self.blur_fc2 = nn.Linear(512, 2)  # Predict blur values for two images

#     def forward(self, image_stack, previous_action=None, mode="Q"):
#         """
#         Forward pass for Q-value prediction or blur value prediction.
#         :param image_stack: Tensor of shape (batch_size, 2, H, W)
#         :param previous_action: Tensor of shape (batch_size, action_dim), required for "Q" mode
#         :param mode: "Q" for Q-value prediction, "blur" for blur value prediction
#         :return: Predicted Q-values or blur values depending on mode
#         """
#         # Shared convolutional feature extractor
#         x = F.relu(self.bn1(self.conv1(image_stack)))
#         x = F.relu(self.bn2(self.conv2(x)))
#         x = F.relu(self.bn3(self.conv3(x)))
#         x = F.relu(self.bn4(self.conv4(x)))
#         x = x.view(x.size(0), -1)  # Flatten the features

#         if mode == "Q":
#             # Q-value prediction
#             assert previous_action is not None, "previous_action is required for Q-value prediction."
#             x = F.relu(self.fc1(x))
#             action_embed = F.relu(self.action_embedding(previous_action))
#             combined = torch.cat([x, action_embed], dim=1)
#             x = F.relu(self.fc2(combined))
#             return self.fc3(x)

#         elif mode == "blur":
#             # Blur value prediction
#             x = F.relu(self.blur_fc1(x))
#             return self.blur_fc2(x)  # Output two blur values

#         else:
#             raise ValueError(f"Invalid mode: {mode}. Choose 'Q' or 'blur'.")

import torch
import torch.nn as nn
import torch.nn.functional as F

class ComplexDQN(nn.Module):
    def __init__(self, input_image_channels, action_size):
        super(ComplexDQN, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(input_image_channels, 64, kernel_size=5, stride=2, padding=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2)
        self.bn3 = nn.BatchNormd(512)
        
        # Shared fully connected layers
        self.shared_fc1 = nn.Linea2d(256)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2r(512 * 16 * 16, 512 * 2 * 2)
        self.shared_fc2 = nn.Linear(512 * 2 * 2, 512)

        # Blur-specific fully connected layer
        self.blur_fc3 = nn.Linear(512, 2)  # Predict blur values for two images

        # Q-value specific layers
        self.fc_reduce = nn.Linear(512, 64)  # Reduce to 64 dimensions for concatenation
        self.action_embedding = nn.Linear(3, 64)
        self.q_fc1 = nn.Linear(64 + 64, 512)
        self.q_fc2 = nn.Linear(512, action_size)

    def freeze_shared_layers(self):
        """
        Freeze the shared layers by setting requires_grad to False.
        """
        for layer in [self.conv1, self.bn1, self.conv2, self.bn2, self.conv3, self.bn3, self.conv4, self.bn4, 
                      self.shared_fc1, self.shared_fc2]:
            for param in layer.parameters():
                param.requires_grad = False

    def forward(self, image_stack, previous_action=None, mode="Q"):
        """
        Forward pass for Q-value prediction or blur value prediction.
        :param image_stack: Tensor of shape (batch_size, 2, H, W)
        :param previous_action: Tensor of shape (batch_size, action_dim), required for "Q" mode
        :param mode: "Q" for Q-value prediction, "blur" for blur value prediction
        :return: Predicted Q-values or blur values depending on mode
        """
        # Shared convolutional feature extractor
        x = F.relu(self.bn1(self.conv1(image_stack)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = x.view(x.size(0), -1)  # Flatten the features

        # Shared fully connected layers
        x = F.relu(self.shared_fc1(x))
        x = F.relu(self.shared_fc2(x))

        if mode == "Q":
            # Q-value prediction
            assert previous_action is not None, "previous_action is required for Q-value prediction."
            reduced_x = F.relu(self.fc_reduce(x))
            action_embed = F.relu(self.action_embedding(previous_action))
            combined = torch.cat([reduced_x, action_embed], dim=1)
            x = F.relu(self.q_fc1(combined))
            return self.q_fc2(x)

        elif mode == "blur":
            # Blur value prediction
            return self.blur_fc3(x)  # Output two blur values

        else:
            raise ValueError(f"Invalid mode: {mode}. Choose 'Q' or 'blur'.")