import torch
import torch.nn as nn


class QNetwork(nn.Module):
    def __init__(self, n_actions: int) -> None:
        super().__init__()
        # conv layers
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        # fully connected layers
        # 3 conv layers reduce 84x84 to 7x7 with 64 filters: 7x7x64 = 3136
        self.fc1 = nn.Linear(in_features=3136, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=n_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        # flatten from (B, 64, 7, 7) to (B, 3136)
        x = x.flatten(start_dim=1)
        x = torch.relu(self.fc1(x))
        # no activation on Q-value output
        x = self.fc2(x)
        return x

