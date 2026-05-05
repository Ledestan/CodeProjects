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

np.random.seed(0)
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
    pca.visualize_pca_face()
    pca.visualize_eigenface_relationship()


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
        self.pca = None
        self.scaler = None
        self.X_scaler = None
        self.explained_var = None # 每个主成分的方差比
        self.cumulative_var = None # 累计方差比

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
        self.scaler = StandardScaler()
        self.X_scaler = self.scaler.fit_transform(self.X)

        # 执行PCA
        n_components = min(150, self.X.shape[1])
        self.pca = PCA(n_components=n_components)
        self.pca.fit(self.X_scaler)

        self.explained_var = self.pca.explained_variance_ratio_ # 获取每个主成分的方差比
        self.cumulative_var = np.cumsum(self.explained_var) # 累计方差比

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

    def visualize_pca_face(self):
        fig, axes = plt.subplots(3, 5, figsize=(10, 6))
        h, w = self.img_size
        
        # 第一行特征脸
        for i, ax in enumerate(axes[0]):
            if i < self.pca.n_components_:
                eigenface = self.pca.components_[i].reshape(h, w)
                ax.imshow(eigenface, cmap="gray")
                ax.set_title(f"PCA {i + 1} ({self.explained_var[i] * 100:.1f}%)")
            ax.set_xticks([])
            ax.set_yticks([])
        axes[0][0].set_ylabel("特征脸", fontsize=10, fontweight='bold')
        
        # 第二行原始人脸
        sample_idx = np.random.choice(self.X.shape[0], min(5, self.X.shape[0]), replace=False)
        for i, ax in enumerate(axes[1]):
            if i < len(sample_idx):
                idx = sample_idx[i]
                original_img = self.X[idx].reshape(h, w)
                ax.imshow(original_img, cmap="gray")
                name = self.Y[idx]
                ax.set_title(f"原始 {name}")
            ax.set_xticks([])
            ax.set_yticks([])
        axes[1][0].set_ylabel("原始脸图像", fontsize=10, fontweight='bold')
            
        # 第三行重构人脸
        x_pca = self.pca.transform(self.X_scaler)
        x_reconstructed = self.pca.inverse_transform(x_pca)
        x_reconstructed_orig_scale = self.scaler.inverse_transform(x_reconstructed)
        for i, ax in enumerate(axes[2]):
            if i < len(sample_idx):
                idx = sample_idx[i]
                reconstructed_img = x_reconstructed_orig_scale[idx].reshape(h, w)
                ax.imshow(reconstructed_img, cmap="gray")
                name = self.Y[idx]
                ax.set_title(f"重构 {name}")
            ax.set_xticks([])
            ax.set_yticks([])
        axes[2][0].set_ylabel("重构脸图像", fontsize=10, fontweight='bold')

        plt.suptitle("PCA 特征脸演示", fontsize=12)
        plt.tight_layout()
        plt.show()

    def visualize_eigenface_relationship(self):
        """特征脸与原始人脸关系可视化"""
        h, w = self.img_size
        n_show_components = 5 # 只用前5个主成分重构
        unique_people = np.unique(self.Y)
        selected_people = np.random.choice(unique_people, size=3, replace=False)

        for person_id in selected_people:
            person_all_idx = np.where(self.Y == person_id)[0]
            sample_indices = person_all_idx[:3] # 每人取3张图

            plt.figure(figsize=(10, 6))
            for row, sample_idx in enumerate(sample_indices):
                # 原始图像
                original_img = self.X[sample_idx].reshape(h, w)

                # PCA投影 + 仅前5个主成分重构
                sample_pca = self.pca.transform(self.X_scaler[sample_idx:sample_idx+1])
                sample_pca_truncated = np.zeros_like(sample_pca)
                sample_pca_truncated[:, :n_show_components] = sample_pca[:, :n_show_components]

                # 原图
                ax1 = plt.subplot(3, 4, row * 4 + 1)
                ax1.imshow(original_img, cmap='gray')
                ax1.set_title(f'人物 - 原图', fontsize=12)
                ax1.axis('off')

                # 重构图
                recon_scaler = self.pca.inverse_transform(sample_pca_truncated)
                recon_img = self.scaler.inverse_transform(recon_scaler).reshape(h, w)
                ax2 = plt.subplot(3, 4, row * 4 + 2)
                ax2.imshow(recon_img, cmap='gray')
                ax2.set_title('重构', fontsize=12)
                ax2.axis('off')

                # 差异图
                diff_img = original_img - recon_img
                ax3 = plt.subplot(3, 4, row * 4 + 3)
                ax3.imshow(diff_img, cmap='RdBu', vmin=-50, vmax=50)
                ax3.set_title('差异图 (红亮/蓝暗)', fontsize=12)
                ax3.axis('off')

                # 权重柱状图
                ax4 = plt.subplot(3, 4, row * 4 + 4)
                weights = sample_pca[0, :n_show_components]
                colors = ['red' if w > 0 else 'blue' for w in weights]
                ax4.bar(range(1, 6), weights, color=colors, alpha=0.7)
                ax4.set_xlabel('主成分')
                ax4.set_ylabel('权重')
                ax4.set_title('投影权重', fontsize=12)
                ax4.grid(True, alpha=0.3)

            plt.suptitle(f'人物{person_id}：特征脸线性组合示意', fontsize=14)
            plt.tight_layout()
            plt.show()


if __name__ == "__main__":
    # pca_iris()
    # pca_face("data/orl_faces", (64, 64))
    pca = PCA_face("data/orl_faces", (64, 64))
    pca.load_face_data()
    pca.data_processing()
    # pca.show_PCA()
    # pca.visualize_pca_face()
    pca.visualize_eigenface_relationship()