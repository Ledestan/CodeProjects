import torch.nn as nn


class DiabetesRegressionModel(nn.Module):
    """用于糖尿病目标值回归的多层感知机（MLP）"""

    def __init__(self, input_dim: int = 10, hidden_dims: tuple = (64, 32)):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], 1),  # 输出 1 个连续值
        )

    def forward(self, x):
        return self.net(x)
