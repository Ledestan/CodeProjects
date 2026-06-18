"""
项目名称: 大学排名聚类分析
创建日期: 2026-06-18
需求文件: data/University.csv

依赖库:
matplotlib>=3.10.8
numpy>=2.2.6
pandas>=3.0.1
scikit-learn>=1.8.0
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.metrics import pairwise_distances
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False


class UniversityAnalysis:
    def __init__(self, path: str):
        """初始化分析器, 加载数据并提取数值特征"""
        self.data = pd.read_csv(path, encoding="utf-8")
        self.feature_cols = self.data.columns[6:].tolist()
        for col in self.feature_cols:
            self.data[col] = pd.to_numeric(self.data[col], errors="coerce")
            self.data[col] = self.data[col].fillna(0.0)

        self.X_scaled = None  # 标准化后的特征矩阵
        self.K_range = range(2, 11)  # 候选 K 值范围
        self.n_clusters = None  # 最优聚类数
        self.dist_matrix = None  # 欧氏距离矩阵(用于 DBSCAN 和评估)
        self.scores = None  # 各算法的轮廓系数评估结果

    def data_preview(self):
        """展示数据基本信息, 描述性统计及多组可视化图表."""
        print("\n" + "=" * 50)
        print("数据基本信息: ")
        self.data.info()
        print("\n" + "=" * 50)
        print(f"描述性统计:\n{self.data[self.feature_cols].describe()}")

        # 选取关键指标绘制直方图
        cols = ["总分", "科学研究", "高端人才"]
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        for i, col in enumerate(cols):
            axes[i].hist(self.data[col], bins=10, edgecolor="black", alpha=0.7)
            axes[i].set(title=col, xlabel=col, ylabel="Count" if i == 0 else None)
        plt.tight_layout()
        plt.show()

        # 箱线图: 整体分布
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        for i, col in enumerate(cols):
            axes[i].boxplot(
                self.data[col],
                vert=True,
                patch_artist=True,
                boxprops=dict(facecolor="lightblue", alpha=0.7),
            )
            axes[i].set_title(col, fontsize=12)
            axes[i].set_ylabel("Value")
        plt.suptitle("Overall Distribution Boxplot (原始尺度)", fontsize=14)
        plt.tight_layout()
        plt.show()

    def data_preprocessing(self):
        """标准化数值特征并计算样本间的欧氏距离矩阵."""
        scaler = StandardScaler()
        self.X_scaled = scaler.fit_transform(self.data[self.feature_cols])
        self.dist_matrix = pairwise_distances(self.X_scaled, metric="euclidean")

    def find_optimal_k(self):
        """综合轮廓系数, 肘部法则和加权评分自动确定最优 K 值"""
        costs, silhouette_scores = [], []
        print("\n" + "=" * 50)
        for k in self.K_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(self.X_scaled)
            costs.append(kmeans.inertia_)
            score = metrics.silhouette_score(self.X_scaled, labels)
            silhouette_scores.append(score)
            print(f"K={k}, 簇内平方和={kmeans.inertia_:.4f}, 轮廓系数={score:.4f}")

        # 绘制肘部法则和轮廓系数图
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
        ax1.plot(self.K_range, costs, "o-")
        ax1.set(
            xticks=self.K_range,
            xlabel="K 值",
            ylabel="簇内平方和",
            title="Elbow Method",
        )
        ax2.plot(self.K_range, silhouette_scores, "o-")
        ax2.set(
            xticks=self.K_range,
            xlabel="K 值",
            ylabel="轮廓系数",
            title="Silhouette Coefficient",
        )
        plt.tight_layout()
        plt.show()

        silhouette_scores = np.array(silhouette_scores)
        costs = np.array(costs)

        # 轮廓系数法: 选择最高轮廓系数的 K
        k_silhouette = self.K_range[np.argmax(silhouette_scores)]
        max_silhouette = max(silhouette_scores)

        # 肘部法则: 二阶差分最大点
        diffs = np.diff(costs)
        second_diffs = np.diff(diffs)
        k_elbow = self.K_range[np.argmax(second_diffs) + 1]

        # 综合评分法: 轮廓系数 (70%) + 成本归一化 (30%)
        silhouette_norm = (silhouette_scores - min(silhouette_scores)) / (
            max(silhouette_scores) - min(silhouette_scores) + 1e-8
        )
        cost_norm = (costs - min(costs)) / (max(costs) - min(costs) + 1e-8)
        combined_scores = 0.7 * silhouette_norm + 0.3 * (1 - cost_norm)
        k_combined = self.K_range[np.argmax(combined_scores)]
        max_combined = max(combined_scores)

        print("\n" + "=" * 50)
        print(f"轮廓系数法最佳 K 值: {k_silhouette} (轮廓系数: {max_silhouette:.4f})")
        print(f"肘部法则最佳 K 值: {k_elbow} (簇内平方和: {costs[k_elbow-2]:.4f})")
        print(f"综合评分法最佳 K 值: {k_combined} (综合分数: {max_combined:.4f})")

        self.n_clusters = k_combined

    def plot_k_distance(self):
        """绘制 K 距离图, 用于确定 DBSCAN 的最优 eps 参数"""
        neighbors = NearestNeighbors(
            n_neighbors=self.n_clusters + 1, metric="precomputed"
        )
        neighbors.fit(self.dist_matrix)
        distances, _ = neighbors.kneighbors(self.dist_matrix)
        k_distances = np.sort(distances[:, -1])[::-1]

        # 通过差分平滑寻找拐点
        diffs = np.diff(k_distances)
        window_size = 5
        smoothed = np.convolve(diffs, np.ones(window_size) / window_size, mode="valid")
        elbow_idx = np.argmax(smoothed) + window_size // 2
        optimal_eps = k_distances[elbow_idx]

        plt.figure(figsize=(8, 6))
        plt.plot(
            range(len(k_distances)), k_distances, marker="o", linestyle="-", color="b"
        )
        plt.axhline(
            y=optimal_eps,
            color="r",
            linestyle="--",
            label=f"Optimal eps ≈ {optimal_eps:.2f}",
        )
        plt.scatter([elbow_idx], [optimal_eps], color="red", zorder=5)
        plt.title(f"K-Distance Graph (k={self.n_clusters})")
        plt.xlabel("Points sorted by distance")
        plt.ylabel(f"Distance to {self.n_clusters}th nearest neighbor")
        plt.grid(True)
        plt.legend()
        plt.show()
        return optimal_eps

    def evaluate(self, labels, name):
        """基于预计算的距离矩阵计算轮廓系数"""
        unique_labels = set(labels)
        if len(unique_labels) < 2:
            return name, None, len(unique_labels)
        if -1 in unique_labels:
            mask = labels != -1
            if mask.sum() == 0:
                return name, None, 0
            valid_labels = labels[mask]
            n_valid_clusters = len(set(valid_labels))
            if n_valid_clusters < 2:
                return name, None, n_valid_clusters
            score = metrics.silhouette_score(
                self.dist_matrix[mask][:, mask], valid_labels, metric="precomputed"
            )
            n_clusters = len(unique_labels) - 1
        else:
            score = metrics.silhouette_score(
                self.dist_matrix, labels, metric="precomputed"
            )
            n_clusters = len(unique_labels)
        return name, score, n_clusters

    def plot_clusters_scatter(self, labels, title="聚类结果"):
        """绘制散点图, 按簇着色"""
        # 获取唯一簇标签并分配颜色
        unique_labels = np.unique(labels)
        # 处理噪声点 (-1) 的颜色为灰色
        cmap = plt.cm.viridis
        colors = [
            cmap(i / max(1, len(unique_labels) - 1)) for i in range(len(unique_labels))
        ]
        # 将标签映射到颜色索引
        label_to_color = {label: colors[i] for i, label in enumerate(unique_labels)}
        # 为每个点生成颜色
        point_colors = [
            label_to_color[label] if label != -1 else (0.5, 0.5, 0.5, 0.7)
            for label in labels
        ]

        plt.figure(figsize=(8, 6))
        plt.scatter(
            self.data["总分"], self.data["科学研究"], c=point_colors, s=60, alpha=0.8
        )
        plt.title(title)
        plt.xlabel("总分 (原始尺度)")
        plt.ylabel("科学研究 (原始尺度)")
        plt.grid(True, linestyle="--", alpha=0.3)

        # 创建图例
        handles = []
        for label in unique_labels:
            if label == -1:
                handles.append(
                    plt.Line2D(
                        [0],
                        [0],
                        marker="o",
                        color="w",
                        markerfacecolor=(0.5, 0.5, 0.5, 0.7),
                        markersize=8,
                        label="Noise",
                    )
                )
            else:
                handles.append(
                    plt.Line2D(
                        [0],
                        [0],
                        marker="o",
                        color="w",
                        markerfacecolor=label_to_color[label],
                        markersize=8,
                        label=f"Cluster {label}",
                    )
                )
        if handles:
            plt.legend(handles=handles, bbox_to_anchor=(1.02, 1), loc="upper left")
        plt.tight_layout()
        plt.show()

    def show_clusters(self):
        """执行 K-means, GMM, 层次聚类和 DBSCAN, 展示聚类可视化并输出各算法的轮廓系数评估"""
        print("\n" + "=" * 50)
        print(f"聚类数: {self.n_clusters}")
        eps_value = self.plot_k_distance()
        print(f"DBSCAN 最优 eps 值: {eps_value:.3f}")

        # K-means
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        labels_kmeans = kmeans.fit_predict(self.X_scaled)
        self.plot_clusters_scatter(
            labels_kmeans, title="K-means 聚类结果 (总分 vs 科学研究)"
        )

        # GMM
        gmm = GaussianMixture(n_components=self.n_clusters, random_state=42)
        labels_gmm = gmm.fit_predict(self.X_scaled)
        self.plot_clusters_scatter(labels_gmm, title="GMM 聚类结果 (总分 vs 科学研究)")

        # 层次聚类
        agg = AgglomerativeClustering(
            n_clusters=self.n_clusters, metric="euclidean", linkage="ward"
        )
        labels_agg = agg.fit_predict(self.X_scaled)
        self.plot_clusters_scatter(labels_agg, title="层次聚类结果 (总分 vs 科学研究)")

        # DBSCAN
        dbscan = DBSCAN(metric="precomputed", eps=eps_value, min_samples=5)
        labels_dbscan = dbscan.fit_predict(self.dist_matrix)
        self.plot_clusters_scatter(
            labels_dbscan, title="DBSCAN 聚类结果 (总分 vs 科学研究)"
        )

        # 统一评估
        self.scores = [
            self.evaluate(labels_kmeans, "K-means"),
            self.evaluate(labels_gmm, "GMM"),
            self.evaluate(labels_agg, "层次聚类"),
            self.evaluate(labels_dbscan, "DBSCAN"),
        ]
        print("\n基于距离矩阵的轮廓系数评估结果:")
        for name, score, n_clusters in self.scores:
            score_str = f"{score:.4f}" if score is not None else "N/A"
            print(f"{name:15s} 轮廓系数: {score_str}, 簇数: {n_clusters}")


if __name__ == "__main__":
    analyzer = UniversityAnalysis("data/University.csv")
    analyzer.data_preview()
    analyzer.data_preprocessing()
    analyzer.find_optimal_k()
    analyzer.show_clusters()
