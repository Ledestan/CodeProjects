"""
PCA 人脸数据集处理

创建日期：2026-04-24
需求文件：data/orl_faces

依赖库：
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

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

DATA_DIR = 'data/orl_faces'
IMG_SIZE = (64, 64)


def load_face_data(path, img_size):
    """
    加载人脸数据集
    path：路径
    img_size：统一的人脸图片尺寸
    return：特征矩阵（样本数，像素数）
            标签（人脸所属于哪个人）
            原始图尺寸
    """
    X, Y = [], []
    # 遍历数据集文件夹
    for person_id in os.listdir(path):
        person_dir = os.path.join(path, person_id)
        if not os.path.isdir(person_dir):
            continue
        # 遍历 person_id 内所有图片，转成灰度图，统一尺寸，展平
        for img_name in os.listdir(person_dir):
            img_path = os.path.join(person_dir, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img_shape = img.shape
            img = cv2.resize(img, img_size) # 统一尺寸
            img_flatten = img.flatten() # 展平
            X.append(img_flatten)
            Y.append(int(person_id[1:]))
    X = np.array(X, dtype=np.float32)
    Y = np.array(Y)

    return X, Y, img_shape

def show_PCA(cumulative_var):
    plt.figure(figsize=(10, 4))
    plt.plot(cumulative_var, 'b--', linewidth=2, label='累计解释方差')
    plt.axhline(y=0.9, color='r', linestyle='--', label='90%')
    plt.axhline(y=0.95, color='g', linestyle='--', label='95%')
    plt.xlabel('主成分的数量')
    plt.ylabel('累计解释方差比例')
    plt.title('人脸 PCA 特征')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 加载数据
    print('加载人脸数据集')
    X, Y, img_shape = load_face_data(DATA_DIR, IMG_SIZE)
    print('人脸数据加载完成')

    # 标准化处理
    scaler = StandardScaler()
    X_scaler = scaler.fit_transform(X)

    # 执行PCA
    print('===== PCA 特征脸分析 =====') 
    n_components = min(150, X.shape[1])
    pca = PCA(n_components=n_components)
    pca.fit(X_scaler)

    explained_var = pca.explained_variance_ratio_ # 获取每个主成分的方差比
    cumulative_var = np.cumsum(explained_var) # 累计方差比
    show_PCA(cumulative_var)