"""
项目名称: 聚类分析
创建日期: 2026-04-02
需求文件: data/Points80.csv, data/Points788.csv

依赖库：
matplotlib>=3.10.8
numpy>=2.2.6
pandas>=3.0.1
scikit-learn>=1.8.0
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.cluster import DBSCAN, KMeans
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False


class ClusterAnalysis:
    def __init__(self, path):
        self.data = pd.read_csv(path, header=None, sep=",")
        self.n_clusters = None
        self.eps = None

    def show_scatter(self, title="Scatter Plot", color="b"):
        """散点图"""
        plt.scatter(self.data.iloc[:, 0], self.data.iloc[:, 1], s=10, c=color)
        plt.xticks([])
        plt.yticks([])
        plt.title(title)
        plt.tight_layout()
        plt.show()

    def elbow_method(self):
        """肘部法则"""
        list_inertia = []
        print("\n" + "=" * 50)
        for i in range(2, 11):
            model = KMeans(n_clusters=i, random_state=0)
            model.fit(self.data)
            inertia = model.inertia_
            print(f"第 {i} 个 K 值，SSE为：{inertia:.4f}")
            list_inertia.append(inertia)

        plt.figure(figsize=(8, 6))
        plt.plot(range(2, 11), list_inertia, "o-")
        plt.xlabel("K Value")
        plt.ylabel("SSE")
        plt.title("Elbow of SSE")
        plt.tight_layout()
        plt.show()

    def silhouette_coefficient(self):
        """轮廓系数"""
        list_silhouette_score = []
        print("\n" + "=" * 50)
        for i in range(2, 11):
            model = KMeans(n_clusters=i, random_state=0)
            model.fit(self.data)
            score = metrics.silhouette_score(self.data, model.labels_)
            print(f"第 {i} 个 K 值，轮廓系数为：{score:.4f}")
            list_silhouette_score.append(score)

        plt.figure(figsize=(8, 6))
        plt.plot(range(2, 11), list_silhouette_score, "o-")
        plt.xlabel("K Value")
        plt.ylabel("轮廓系数")
        plt.title("Silhouette Coefficient")
        plt.tight_layout()
        plt.show()

        print("\n" + "=" * 50)
        self.n_clusters = list_silhouette_score.index(max(list_silhouette_score)) + 2
        print(f"最佳 K 值：{self.n_clusters}")

    def show_KMeans(self):
        """K-Means (K 均值) 聚类"""
        model = KMeans(n_clusters=self.n_clusters, random_state=0)
        model.fit(self.data)

        print("K-Means 聚类结果：\n", model.labels_)
        print("K-Means 聚类中心：\n", model.cluster_centers_)

        plt.figure(figsize=(8, 6))
        plt.scatter(
            model.cluster_centers_[:, 0],
            model.cluster_centers_[:, 1],
            marker="*",
            c="r",
        )
        self.show_scatter("K-Means 聚类结果散点图", color=model.labels_)

        # 模型评估
        print(f"轮廓系数：{metrics.silhouette_score(self.data, model.labels_)}")
        print(
            f"戴维斯-布尔丁指数：{metrics.davies_bouldin_score(self.data, model.labels_)}"
        )
        print(
            f"卡林斯基-哈拉巴斯指数：{metrics.calinski_harabasz_score(self.data, model.labels_)}"
        )

    def show_GMM(self):
        """GMM (高斯混合模型聚类)"""
        plt.figure(figsize=(8, 6))
        model = GaussianMixture(
            n_components=self.n_clusters, covariance_type="full", random_state=0
        )
        model.fit(self.data)
        plt.scatter(model.means_[:, 0], model.means_[:, 1], marker="*", s=50, c="r")
        self.show_scatter("GMM 聚类结果散点图", color=model.predict(self.data))

    def plot_k_distance(self, k=4):
        """k 距离图"""
        neighbors = NearestNeighbors(n_neighbors=k + 1)
        neighbors_fit = neighbors.fit(self.data)
        distances, _ = neighbors_fit.kneighbors(self.data)
        k_distances = distances[:, -1]
        k_distances_sorted = np.sort(k_distances)[::-1]

        # 自动拐点检测（一阶差分最大值法）
        diffs = np.diff(k_distances_sorted)
        elbow_idx = np.argmax(diffs)  # 最陡峭位置
        self.eps = k_distances_sorted[elbow_idx]

        plt.figure(figsize=(8, 6))
        plt.plot(range(len(k_distances_sorted)), k_distances_sorted, "o-", color="b")
        plt.axvline(
            x=elbow_idx, linestyle="--", color="r", label=f"eps ≈ {self.eps:.3f}"
        )
        plt.title(f"K-Distance Graph (k={k})")
        plt.xlabel("Points sorted by distance")
        plt.ylabel(f"Distance to {k}th nearest neighbor")
        plt.grid(True)
        plt.legend()
        plt.show()

    def show_DBSCAN(self):
        """DBSCAN (基于密度的带噪声应用空间聚类)"""
        plt.figure(figsize=(12, 6))
        minPoints = list(range(1, 8))
        for i, min_samples in enumerate(minPoints):
            model = DBSCAN(eps=self.eps, min_samples=min_samples)
            labels = model.fit_predict(self.data)
            plt.subplot(2, 4, i + 1)
            plt.scatter(self.data.iloc[:, 0], self.data.iloc[:, 1], s=10, c=labels)
            plt.xticks([])
            plt.yticks([])
            plt.title(f"DBSCAN (min_samples={min_samples})")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    if input("Input not null: Points80.csv, if null: Points788.csv\n"):
        ca = ClusterAnalysis("data/Points80.csv")
    else:
        ca = ClusterAnalysis("data/Points788.csv")
    ca.show_scatter()
    ca.elbow_method()
    ca.silhouette_coefficient()
    ca.show_KMeans()
    ca.show_GMM()
    ca.plot_k_distance()
    ca.show_DBSCAN()
