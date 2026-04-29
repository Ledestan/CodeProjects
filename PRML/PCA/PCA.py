"""
PCA 数据集处理

创建日期: 2026-04-23
需求文件: data/orl_faces

依赖库:
numpy>=2.2.6
opencv-python>=4.12.0.88
matplotlib>=3.10.8
scikit-learn>=1.8.0
"""

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from sklearn.datasets import load_iris

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False


def pca_iris():
    pca = PCA_Iris
    pca.data_processing()
    pca.show_ratio()

def pca_face(path, img_size):
    pca = PCA_face(path, img_size)
    pca.load_face_data()
    pca.data_processing()
    pca.show_PCA()


class PCA_Iris:
    def __init__(self):
        self.pca = PCA()
    
    def data_processing(self):
        """数据处理"""
        iris = load_iris()
        X, y = iris.data, iris.target
        scaler = StandardScaler() # 初始化标准化器
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
        var_exp = self.pca.explained_variance_ # 主成分方差

        # 可视化
        fig, ax1 = plt.subplots(1, 1, figsize=(8, 4))
        ax1.set_xlabel("主成分编号")
        ax1.set_ylabel("主成分贡献率")
        ax1.bar(np.arange(len(var_exp)), self.pca.explained_variance_ratio_, width=0.5, align="center")
        ax1.set_xlim([-1, len(var_exp)])

        ax2 = ax1.twinx()
        ax2.set_ylabel("主成分累计贡献率")
        ax2.set_ylim([0, 1.2])
        ax2.step(np.arange(len(var_exp)), np.cumsum(self.pca.explained_variance_ratio_), where="mid", color="g")
        ax2.axhline(y=0.9, color="r", linestyle="--")

        plt.tight_layout()
        plt.show()


class PCA_face:
    def __init__(self, path, img_size):
        self.path = path
        self.img_size = img_size
        self.X = None
        self.Y = None
        self.img_shape = None
        self.cumulative_var = None

    def load_face_data(self):
        """加载人脸数据集"""
        X, Y = [], []
        # 遍历数据集文件夹
        for person_id in os.listdir(self.path):
            person_dir = os.path.join(self.path, person_id)
            if not os.path.isdir(person_dir):
                continue
            # 遍历 person_id 内所有图片，转成灰度图，统一尺寸，展平
            for img_name in os.listdir(person_dir):
                img_path = os.path.join(person_dir, img_name)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                self.img_shape = img.shape
                img = cv2.resize(img, self.img_size) # 统一尺寸
                img_flatten = img.flatten() # 展平
                X.append(img_flatten)
                Y.append(int(person_id[1:]))
        self.X = np.array(X, dtype=np.float32)
        self.Y = np.array(Y)

    def data_processing(self):
        """数据处理"""
        # 标准化处理
        scaler = StandardScaler()
        X_scaler = scaler.fit_transform(self.X)

        # 执行PCA
        n_components = min(150, self.X.shape[1])
        pca = PCA(n_components=n_components)
        pca.fit(X_scaler)

        explained_var = pca.explained_variance_ratio_ # 获取每个主成分的方差比
        self.cumulative_var = np.cumsum(explained_var) # 累计方差比

    def show_PCA(self):
        """PCA 结果展示"""
        plt.figure(figsize=(10, 4))
        plt.plot(self.cumulative_var, "b--", linewidth=2, label="累计解释方差")
        plt.axhline(y=0.9, color="r", linestyle="--", label="90%")
        plt.axhline(y=0.95, color="g", linestyle="--", label="95%")
        plt.xlabel("主成分的数量")
        plt.ylabel("累计解释方差比例")
        plt.title("人脸 PCA 特征")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    pca_iris()
    pca_face("data/orl_faces", (64, 64))