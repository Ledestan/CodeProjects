"""
PCA 数据集处理

创建日期：2026-04-23

依赖库：
numpy>=2.2.6
matplotlib>=3.10.8
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from sklearn.datasets import load_iris

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


def pca_iris():
    iris = load_iris()
    x, y = iris.data, iris.target
    scaler = StandardScaler() # 初始化标准化器
    x_scaler = scaler.fit_transform(x)
    
    pca = PCA()
    x_pca = pca.fit_transform(x_scaler)

    target_name = ['山鸢尾', '变色鸢尾', '维吉尼亚鸢尾']
    plt.figure(figsize=(8, 4))
    for i, name in enumerate(target_name):
        plt.scatter(x_pca[y == i, 0], x_pca[y == i, 1],
                    s=8, alpha=0.8, label=name)
    plt.legend()
    plt.xlabel('PCA1', fontsize=10)
    plt.ylabel('PCA2', fontsize=10)
    plt.title('Iris 数据的 PCA 结果')
    plt.tight_layout()
    plt.show()

    return pca


def show_ratio(pca):
    var_exp = pca.explained_variance_ # 主成分方差

    # 可视化
    fig, ax1 = plt.subplots(1, 1, figsize=(8, 4))
    ax1.set_xlabel('主成分编号')
    ax1.set_ylabel('主成分贡献率')
    ax1.bar(np.arange(len(var_exp)), pca.explained_variance_ratio_, width=0.5, align='center')
    ax1.set_xlim([-1, len(var_exp)])

    ax2 = ax1.twinx()
    ax2.set_ylabel('主成分累计贡献率')
    ax2.set_ylim([0, 1.2])
    ax2.step(np.arange(len(var_exp)), np.cumsum(pca.explained_variance_ratio_), where='mid', color='g')
    ax2.axhline(y=0.9, color='r', linestyle='--')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    pca = pca_iris()
    show_ratio(pca)