"""
项目名称: Iris 降维
创建日期: 2026-04-23
需求文件: 无

依赖库:
matplotlib>=3.10.8
numpy>=2.2.6
scikit-learn>=1.8.0
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

np.random.seed(0)
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False


class IrisDimReduction:
    def __init__(self):
        self.pca = PCA()

    def data_processing(self):
        """数据处理"""
        iris = load_iris()
        X, y = iris.data, iris.target
        scaler = StandardScaler()  # 初始化标准化器
        X_scaler = scaler.fit_transform(X)

        X_pca = self.pca.fit_transform(X_scaler)

        target_name = ["山鸢尾", "变色鸢尾", "维吉尼亚鸢尾"]
        plt.figure(figsize=(8, 4))
        for i, name in enumerate(target_name):
            plt.scatter(X_pca[y == i, 0], X_pca[y == i, 1], s=8, alpha=0.8, label=name)
        plt.legend()
        plt.xlabel("PCA1", fontsize=10)
        plt.ylabel("PCA2", fontsize=10)
        plt.title("Iris 数据的 PCA 结果")
        plt.tight_layout()
        plt.show()

    def show_ratio(self):
        """结果展示"""
        var_exp = self.pca.explained_variance_  # 主成分方差

        # 可视化
        fig, ax1 = plt.subplots(1, 1, figsize=(8, 4))
        ax1.set_xlabel("主成分编号")
        ax1.set_ylabel("主成分贡献率")
        ax1.bar(
            np.arange(len(var_exp)),
            self.pca.explained_variance_ratio_,
            width=0.5,
            align="center",
        )
        ax1.set_xlim([-1, len(var_exp)])

        ax2 = ax1.twinx()
        ax2.set_ylabel("主成分累计贡献率")
        ax2.set_ylim([0, 1.2])
        ax2.step(
            np.arange(len(var_exp)),
            np.cumsum(self.pca.explained_variance_ratio_),
            where="mid",
            color="g",
        )
        ax2.axhline(y=0.9, color="r", linestyle="--")

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    pca = IrisDimReduction()
    pca.data_processing()
    pca.show_ratio()
