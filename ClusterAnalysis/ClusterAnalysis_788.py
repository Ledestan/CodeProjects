"""
聚类分析工具

创建日期: 2026-04-09
需求文件: data\points788.txt

依赖库：
pandas>=3.0.1
numpy>=2.2.6
matplotlib>=3.10.8
scikit-learn>=1.8.0
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False


class ClusterAnalysis_788:
    def __init__(self, path:str):
        self.data = pd.read_csv(path, header=None, sep=",")
    
    def show_scatter(self, title="Scatter Plot", color="b"):
        """散点图"""
        plt.scatter(self.data.iloc[:, 0], self.data.iloc[:, 1], s=10, c=color)
        # plt.axis("off") # 隐藏坐标轴
        plt.xticks([]) # 隐藏 x 轴数据
        plt.yticks([]) # 隐藏 y 轴数据
        plt.title(title)
        plt.tight_layout()

    def display(self):
        """合并预览 K-Means, GMM, DBSCAN"""
        plt.figure(figsize=(9, 3))
        for i, method in enumerate([self.show_scatter, self.show_KMeans, self.show_GMM]):
            plt.subplot(1, 3, i + 1)
            method()
        plt.tight_layout()
        plt.show()

    def show_KMeans(self):
        """K 均值聚类"""
        model = KMeans(n_clusters=7, random_state=28)
        model.fit(self.data)
        self.show_scatter("K-Meams 聚类结果散点图", color=model.labels_)
        plt.scatter(model.cluster_centers_[:, 0], model.cluster_centers_[:, 1], marker="*", s=50, c="r")

    def show_GMM(self):
        """高斯混合模型聚类"""
        model = GaussianMixture(n_components=7, covariance_type="full", random_state=28)
        model.fit(self.data)
        self.show_scatter("GMM 聚类结果散点图", color=model.predict(self.data))
        plt.scatter(model.means_[:, 0], model.means_[:, 1], marker="*", s=50, c="r")

    def show_DBSCAN(self):
        """基于密度的带噪声应用空间聚类"""
        minPoints = list(range(1, 8))
        plt.figure(figsize=(12, 6))
        for i, min_samples in enumerate(minPoints):
            model = DBSCAN(eps=0.5, min_samples=min_samples)
            labels = model.fit_predict(self.data)
            plt.subplot(2, 4, i + 1)
            self.show_scatter(title=f"DBSCAN (min_samples={min_samples})", color=labels)
        plt.tight_layout()
        plt.show()

    def plot_k_distance(self, k=None):
        """k 距离图"""
        neighbors = NearestNeighbors(n_neighbors=k+1)
        neighbors_fit = neighbors.fit(self.data) # 计算距离
        distances, indices = neighbors_fit.kneighbors(self.data)
        
        k_distances = distances[:, -1] # 提取距离
        k_distances_sorted = np.sort(k_distances)[::-1] # 降序排列
        
        plt.figure(figsize=(8, 6))
        plt.plot(range(len(k_distances_sorted)), k_distances_sorted, marker="o", linestyle="-", color="b")
        plt.title(f"K-Distance Graph (k={k})")
        plt.xlabel("Points sorted by distance")
        plt.ylabel(f"Distance to {k}th nearest neighbor")
        plt.grid(True)
        plt.show()

if __name__ == "__main__":
    ca = ClusterAnalysis_788("data/points788.txt")
    ca.display()
    ca.plot_k_distance(4)
    ca.show_DBSCAN()