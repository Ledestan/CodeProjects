"""
项目名称: 线性回归
创建日期: 2026-05-29

需求文件: data

依赖库:
matplotlib>=3.10.9
numpy>=2.2.6
pandas>=3.0.1
scikit-learn>=1.8.0
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False


class LinearRegressionGD:
    """使用梯度下降求解的线性回归"""

    def __init__(self, path: str):
        self.data = pd.read_csv(
            path, sep=",", header=None, names=["population", "profit"]
        )
        self.x = self.data["population"].values.reshape(-1, 1)  # 特征矩阵
        self.y = self.data["profit"].values.reshape(-1, 1)  # 目标向量
        self.X_train = None  # 训练集特征 (含截距列)
        self.y_train = None  # 训练集目标
        self.theta = (
            None  # 模型参数 (权重向量), 形状应为 (2, 1) 或 (1, 2), 对应截距和斜率
        )
        self.lr = 0.001  # 学习率 (Learning Rate)
        self.epochs = 200  # 迭代次数

    def data_explore(self):
        """数据探索"""
        print("\n" + "=" * 50)
        print(f"数据基本信息:")
        print(self.data.info())
        print(f"统计描述:\n{self.data.describe()}")

        # 盒图（箱线图）
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        # 人口特征的盒图
        axes[0].boxplot(self.data["population"])
        axes[0].set_title("人口 (Population) 盒图")
        axes[0].set_ylabel("人口")
        # 利润目标的盒图
        axes[1].boxplot(self.data["profit"])
        axes[1].set_title("利润 (Profit) 盒图")
        axes[1].set_ylabel("利润")
        plt.tight_layout()
        plt.show()

        # 散点图
        plt.scatter(self.x, self.y)
        plt.title("人口 - 利润散点图")
        plt.xlabel("人口")
        plt.ylabel("利润")
        plt.tight_layout()
        plt.show()

    def sklearn_fit_and_eval(self):
        """使用 sklearn 进行线性回归并打印评估结果"""
        # 划分训练集和测试集
        x_train, x_test, y_train, y_test = train_test_split(
            self.x, self.y, test_size=0.2, random_state=42
        )

        # SKlearn 线性回归建模与评估
        model = LinearRegression()
        model.fit(x_train, y_train)
        print("\n" + "=" * 50)
        print(f"回归系数: {model.coef_}")
        print(f"截距: {model.intercept_}")
        print(f"模型得分 (测试集): {model.score(x_test, y_test)}")
        print(f"模型得分 (训练集): {model.score(x_train, y_train)}")
        print(f"均方误差: {mean_squared_error(y_test, model.predict(x_test))}")

        # 梯度下降数据
        self.X_train = np.c_[np.ones(len(x_train)), x_train]  # (77, 2)
        self.y_train = y_train.flatten()  # (77,)
        self.theta = np.zeros(self.X_train.shape[1])  # (2,)

    def gradient_descent(self):
        """使用梯度下降训练线性回归模型"""
        if self.X_train is None or self.y_train is None:
            raise RuntimeError("请先调用 sklearn_fit_and_eval() 初始化训练数据。")

        m, n = self.X_train.shape  # m=77, n=2
        history = []

        for i in range(self.epochs):
            h = np.dot(self.X_train, self.theta)  # 预测值 (77,)
            loss = h - self.y_train  # 误差 (77,)，使用训练集目标
            grad = np.dot(self.X_train.T, loss) / m  # 梯度 (2,)
            self.theta -= self.lr * grad  # 更新参数，使用 self.lr
            cost = (1 / (2 * m)) * np.sum(loss**2)
            history.append(cost)

        plt.plot(range(len(history)), history, label=f"学习率: {self.lr}")
        plt.title("损失函数曲线")
        plt.xlabel("迭代次数")
        plt.ylabel("损失函数")
        plt.legend()
        plt.show()

        print("\n梯度下降结果:")
        print(f"截距 (theta0): {self.theta[0]:.4f}")
        print(f"斜率 (theta1): {self.theta[1]:.4f}")


if __name__ == "__main__":
    model = LinearRegressionGD("data/ex1data1.txt")
    model.data_explore()
    model.sklearn_fit_and_eval()
    model.gradient_descent()
