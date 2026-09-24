"""
项目名称: 糖尿病预测
创建时间: 2026/09/21
"""

import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_diabetes
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["axes.unicode_minus"] = False


def load_data(path: str = "data/diabetes.csv"):
    """从 Sklearn 加载糖尿病数据集，失败则回退到本地 CSV"""
    try:
        print("尝试从 Sklearn 加载数据...")
        diabetes = load_diabetes()
        df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
        df["target"] = diabetes.target
        return df
    except Exception as e1:
        print(f"从 Sklearn 加载失败: {e1}")
        try:
            print("尝试从本地文件加载...")
            # 获取本文件所在路径
            base_dir = os.path.dirname(os.path.abspath(__file__))
            file_path = os.path.join(base_dir, path)
            df = pd.read_csv(file_path)
            return df
        except Exception as e2:
            print(f"数据加载失败: {e2}")
            return None


def explore_data(df):
    """打印数据集基本信息"""
    print("\n" + "=" * 50)
    print(f"数据形状: {df.shape}")
    print("\n" + "=" * 50)
    print("数据信息: ")
    df.info()
    print("\n" + "=" * 50)
    print("数据描述: ")
    print(df.describe())


def plot_correlation(df):
    """绘制特征相关性热力图"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
    plt.tight_layout()
    plt.show()


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


class Trainer:
    """糖尿病回归训练器"""

    def __init__(self, batch_size: int = 16, lr: float = 0.005):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 数据加载
        diabetes = load_data()
        X = diabetes.iloc[:, :-1].values
        y = diabetes["target"].values

        # 划分训练集与测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
        )

        # 转换为 PyTorch 张量
        self.X_train_t = torch.tensor(X_train, dtype=torch.float32)
        self.y_train_t = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
        self.X_test_t = torch.tensor(X_test, dtype=torch.float32)
        self.y_test_t = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)

        # 创建DataLoader（批量训练）
        self.train_loader = DataLoader(
            TensorDataset(self.X_train_t, self.y_train_t),
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
        )
        self.test_loader = DataLoader(
            TensorDataset(self.X_test_t, self.y_test_t),
            batch_size=batch_size,
            shuffle=False,
        )

        # 初始化模型、损失函数、优化器
        self.model = DiabetesRegressionModel(input_dim=self.X_train_t.shape[1])
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def train_one_epoch(self):
        """训练一个 epoch，返回平均 MSE"""
        self.model.train()
        total_loss = 0.0
        for xb, yb in self.train_loader:
            self.optimizer.zero_grad()
            loss = self.criterion(self.model(xb), yb)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item() * xb.size(0)
        return total_loss / len(self.train_loader.dataset)

    @torch.no_grad()
    def evaluate(self):
        """在测试集上评估，返回平均 MSE"""
        self.model.eval()
        total_loss = 0.0
        for xb, yb in self.test_loader:
            total_loss += self.criterion(self.model(xb), yb).item() * xb.size(0)
        return total_loss / len(self.test_loader.dataset)

    def fit(self, epochs: int = 300):
        """训练模型，返回每个 epoch 的训练损失"""
        train_losses = []
        print("开始训练...")
        for epoch in range(epochs):
            train_loss = self.train_one_epoch()
            train_losses.append(train_loss)
            if (epoch + 1) % 20 == 0:
                test_loss = self.evaluate()
                print(
                    f"Epoch [{epoch + 1}/{epochs}] | "
                    f"train MSE: {train_loss:.4f} | test MSE: {test_loss:.4f}"
                )
        return train_losses

    @torch.no_grad()
    def final_report(self):
        """训练结束后计算训练/测试集的 MSE 与 R²，返回 (y_true, y_pred, test_r2)"""
        self.model.eval()
        y_train_pred = self.model(self.X_train_t)
        y_test_pred = self.model(self.X_test_t)

        train_mse = self.criterion(y_train_pred, self.y_train_t).item()
        test_mse = self.criterion(y_test_pred, self.y_test_t).item()
        train_r2 = r2_score(self.y_train_t, y_train_pred.numpy())
        test_r2 = r2_score(self.y_test_t, y_test_pred.numpy())

        print("\n" + "=" * 50)
        print("评估结果")
        print(f"训练集 MSE: {train_mse:.4f}, R²: {train_r2:.4f}")
        print(f"测试集 MSE: {test_mse:.4f}, R²: {test_r2:.4f}")

        return self.y_test_t.numpy(), y_test_pred.cpu().numpy(), test_r2


def plot_loss_curve(train_losses):
    """绘制训练损失曲线"""
    plt.figure(figsize=(6, 4))
    plt.plot(train_losses)
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Training Loss Curve")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_predictions(y_true, y_pred, r2):
    """绘制测试集真实值对比预测值的散点图"""
    plt.figure(figsize=(6, 4))
    plt.scatter(y_true, y_pred, alpha=0.6)
    plt.plot(
        [y_true.min(), y_true.max()],
        [y_true.min(), y_true.max()],
        "r--",
    )
    plt.xlabel("True Values")
    plt.ylabel("Predictions")
    plt.title(f"Test Set: True vs Predicted ($R^2$ = {r2:.3f})")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # 数据探索
    df = load_data()
    if df is not None:
        explore_data(df)
        plot_correlation(df)

    # 训练与评估
    trainer = Trainer(batch_size=16, lr=0.005)
    train_losses = trainer.fit(epochs=300)

    # 绘制训练损失曲线
    plot_loss_curve(train_losses)

    # 最终评估 + 散点图
    y_test, y_test_pred, test_r2 = trainer.final_report()
    plot_predictions(y_test, y_test_pred, test_r2)