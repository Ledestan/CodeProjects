"""
人脸识别分析

创建日期: 2026-05-14
需求文件: data

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

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False


class FaceRecAnalysis:
    def __init__(self, path, img_size):
        self.path = path
        self.img_size = img_size
        self.img_shape = (self.img_size[1], self.img_size[0]) # 数据形状
        self.X = None # 原始图像数据矩阵
        self.y = None # 标签向量
        self.scaler = None # 标准化器
        self.pca = None # PCA 模型
        self.X_scaler = None # 标准化后的数据
        self.explained_var = None # 主成分方差比
        self.cumulative_var = None # 累计方差比
        self.X_pca = None # 降维坐标

    def load_data(self):
        """加载 ORL 人脸数据集并预处理"""
        X, y = [], []
        # 遍历数据集文件夹
        for person_id in os.listdir(self.path):
            person_dir = os.path.join(self.path, person_id)
            if not os.path.isdir(person_dir):
                continue
            # 遍历 person_id 内所有图片, 转成灰度图, 统一尺寸, 展平
            for img_name in os.listdir(person_dir):
                img_path = os.path.join(person_dir, img_name)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                img = cv2.resize(img, self.img_size) # 统一尺寸
                img_flatten = img.flatten() # 展平
                X.append(img_flatten)
                y.append(int(person_id[1:]))
        self.X = np.array(X, dtype=np.float32)
        self.y = np.array(y)

        # 标准化处理
        self.scaler = StandardScaler()
        self.X_scaler = self.scaler.fit_transform(self.X)

    def pca_analyze(self):
        """PCA 分析"""
        # 执行PCA
        n_components = min(150, self.X.shape[1])
        self.pca = PCA(n_components=n_components)
        self.pca.fit(self.X_scaler)

        self.explained_var = self.pca.explained_variance_ratio_ # 获取每个主成分的方差比
        self.cumulative_var = np.cumsum(self.explained_var) # 累计方差比

        self.X_pca = self.pca.transform(self.X_scaler)  # 降维
    
        # 提取特征脸
        self.eigenfaces = [comp.reshape(self.img_shape) for comp in self.pca.components_]

    def analyze_reconstruction_error(self):
        """重构误差分析"""

    def lda_analyze(self):
        """LDA 分析"""

    def mixed_analyze(self):
        """混合分析 PCA + LDA"""

    def comparative_analyze(self):
        """对比分析结果"""


if __name__ == "__main__":
    FaceRec = FaceRecAnalysis("data", (64, 64))