"""
聚类分析工具

创建日期: 2026-04-02
需求文件: Data/points80.txt, Data/points788.txt

依赖库：
pandas>=3.0.1
numpy>=2.2.6
matplotlib>=3.10.8
scikit-learn>=1.8.0
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False


def points_80(path):
    ca = ClusterAnalysis_80(path)
    ca.show_scatter()
    ca.elbow_method()
    ca.silhouette_coefficient()

def points_788(path):
    ca = ClusterAnalysis_788(path)
    ca.display()
    ca.plot_k_distance(4)
    ca.show_DBSCAN()


class ClusterAnalysis_80:
    def __init__(self, path:str):
        self.data = pd.read_csv(path, header=None, sep="\t")

    def show_scatter(self, title="Scatter Plot", color="b"):
        """散点图"""
        plt.scatter(self.data.iloc[:, 0], self.data.iloc[:, 1], c=color)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title(title)
        plt.tight_layout()
        plt.show()

    def elbow_method(self):
        """肘部法则"""
        list_inertia = []
        for i in range(2, 11):
            model = KMeans(n_clusters=i, random_state=0)
            model.fit(self.data)
            list_inertia.append(model.inertia_)
        plt.plot(range(2, 11), list_inertia, "o-")
        plt.xlabel("K Value")
        plt.ylabel("SSE")
        plt.title("Elbow of SSE")
        plt.tight_layout()
        plt.show()

    def silhouette_coefficient(self):
        """轮廓系数"""
        list_silhouette_score = []
        print("\n" + "=" * 50 )
        for i in range(2, 11):
            model = KMeans(n_clusters=i, random_state=0)
            model.fit(self.data)
            score = metrics.silhouette_score(self.data, model.labels_)
            print(f"第 {i} 个 K 值，轮廓系数为：{score:.4f}")
            list_silhouette_score.append(score)
        plt.plot(range(2, 11), list_silhouette_score, "o-")
        plt.xlabel("K Value")
        plt.ylabel("轮廓系数")
        plt.title("Silhouette Coefficient")
        plt.tight_layout()
        plt.show()

        # K-Means 聚类效果评估
        print("\n" + "=" * 50 )
        best_k = list_silhouette_score.index(max(list_silhouette_score)) + 2
        print(f"最佳 K 值：{best_k}")

        model = KMeans(n_clusters=best_k, random_state=0)
        model.fit(self.data)
        print("K-Meams 聚类结果：\n", model.labels_)
        print("K-Meams 聚类中心：\n", model.cluster_centers_)

        plt.scatter(model.cluster_centers_[:, 0], model.cluster_centers_[:, 1], marker="*", c="r")
        self.show_scatter("K-Meams 聚类结果散点图", model.labels_)

        # 模型评估
        print(f"轮廓系数：{metrics.silhouette_score(self.data, model.labels_)}")
        print(f"戴维斯-布尔丁指数：{metrics.davies_bouldin_score(self.data, model.labels_)}")
        print(f"卡林斯基-哈拉巴斯指数：{metrics.calinski_harabasz_score(self.data, model.labels_)}")


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
    points_80("Data/points80.txt")
    points_788("Data/points788.txt")