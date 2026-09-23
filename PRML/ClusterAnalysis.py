"""
项目名称: 聚类分析
创建日期: 2026-04-02

需求文件:
- data/Points80.csv
- data/Points788.csv
- data/MallCustomers.csv
- data/University.csv

依赖库：
matplotlib>=3.10.8
numpy>=2.2.6
pandas>=3.0.1
seaborn>=0.13.2
kmodes>=0.12.2
scipy>=1.17.1
scikit-learn>=1.8.0
"""

from math import pi

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from kmodes.kprototypes import KPrototypes
from scipy.spatial.distance import cdist
from sklearn import metrics
from sklearn.cluster import DBSCAN, OPTICS, AgglomerativeClustering, KMeans
from sklearn.metrics import pairwise_distances
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams.update(
    {
        "axes.facecolor": (0, 0, 0, 0),  # 设置背景透明
        "axes.edgecolor": "black",  # 保留边框颜色
        "figure.facecolor": "white",  # 画布背景为白色
        "legend.facecolor": "white",  # 图例背景为白色
    }
)


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


class CustomerAnalysis:
    def __init__(self, path: str):
        self.path = path
        self.data = pd.read_csv(path, sep=",")
        self.data.drop("CustomerID", axis=1, inplace=True)
        self.data.columns = ["Gender", "Age", "Income", "Score"]
        self.data_raw = self.data.copy()  # 保留原始尺度数据用于可视化
        self.K_range = range(2, 11)
        self.n_clusters = None
        self.data_array = None  # data array
        self.X_num = None  # data 数据部分
        self.X_cat = None  # data 类别部分
        self.dist_num = None  # data 数据部分距离矩阵
        self.dist_cat = None  # data 类别部分距离矩阵
        self.gamma = 0.5  # 性别特征权重
        self.dist_mixed = None  # data 距离混合距离
        self.scores = None  # 聚类结果评估

    def data_preview(self):
        print("\n" + "=" * 50)
        print(f"数据基本信息: ")
        self.data.info()
        print("\n" + "=" * 50)
        print(f"描述性统计:\n{self.data.describe()}")
        print(self.data[["Gender"]].describe().T)

        cols = ["Age", "Income", "Score"]

        # 展示 Age, Income, Score 直方图
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        for i, col in enumerate(cols):
            axes[i].hist(self.data[col], bins=10, edgecolor="black")
            axes[i].set(title=col, xlabel=col, ylabel="Count" if i == 0 else None)
        plt.tight_layout()
        plt.show()

        # 盒图: 展示 Age, Income, Score 的整体分布及性别对比
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

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        for i, col in enumerate(cols):
            male_data = self.data[self.data["Gender"] == "Male"][col]
            female_data = self.data[self.data["Gender"] == "Female"][col]
            box_data = [male_data, female_data]
            bp = axes[i].boxplot(
                box_data,
                patch_artist=True,
                tick_labels=["Male", "Female"],
                boxprops=dict(alpha=0.7),
                medianprops=dict(color="black"),
            )
            bp["boxes"][0].set_facecolor("#4682B4")
            bp["boxes"][1].set_facecolor("#FFC0CB")
            axes[i].set_title(col, fontsize=12)
            axes[i].set_ylabel("Value")
        plt.suptitle("Distribution by Gender (Boxplot)", fontsize=14)
        plt.tight_layout()
        plt.show()

        # 特征与性别的散点图矩阵
        g = sns.pairplot(
            self.data,
            vars=["Age", "Income", "Score"],
            hue="Gender",
            palette=["#4682B4", "#FFC0CB"],
            diag_kind="kde",
            plot_kws={"alpha": 0.6, "s": 40, "edgecolor": "none"},
            diag_kws={"alpha": 0.6, "linewidth": 1.5},
        )
        g.add_legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0, title="Gender")
        plt.subplots_adjust(top=0.95, right=0.85)
        plt.suptitle("Feature Distributions & Relationships", fontsize=16, y=0.98)
        plt.show()

    def data_preprocessing(self):
        """数据预处理"""
        # 数据标准化
        cols = ["Age", "Income", "Score"]
        scaler = StandardScaler()
        self.data[cols] = scaler.fit_transform(self.data[cols])

        self.data_array = (
            self.data.values
        )  # DataFrame 转 array, KPrototypes 需要 array 输入

        # 提取数值特征矩阵和类别特征向量
        self.X_num = self.data_array[:, 1:].astype(float)  # 数值部分
        self.X_cat = self.data_array[:, 0]  # 类别部分

        # 数值欧氏距离 + 类别汉明距离 -> 混合距离矩阵
        self.dist_num = cdist(self.X_num, self.X_num, metric="euclidean")
        self.dist_cat = (self.X_cat[:, np.newaxis] != self.X_cat[np.newaxis, :]).astype(
            float
        )
        self.dist_mixed = self.dist_num + self.gamma * self.dist_cat

    def find_optimal_k(self):
        """综合多种指标自动确定最佳 K 值"""
        # K‑Prototypes 训练
        costs, silhouette_scores = [], []
        print("\n" + "=" * 50)
        for i in self.K_range:
            model = KPrototypes(
                n_clusters=i, init="Cao", verbose=0, gamma=self.gamma, random_state=0
            )
            model.fit(self.data_array, categorical=[0])
            costs.append(model.cost_)
            score = metrics.silhouette_score(
                self.dist_mixed, model.labels_, metric="precomputed"
            )
            silhouette_scores.append(score)
            print(f"K={i}, 簇内平方和={model.cost_:.4f}, 轮廓系数={score:.4f}")

        # 肘部法则和轮廓系数图展示
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
        ax1.plot(self.K_range, costs, "o-")
        ax1.set(
            xticks=self.K_range,
            xlabel="K Value",
            ylabel="Cost (混合误差)",
            title="Elbow Method (K-Prototypes Cost)",
        )
        ax2.plot(self.K_range, silhouette_scores, "o-")
        ax2.set(
            xticks=self.K_range,
            xlabel="K Value",
            ylabel="轮廓系数",
            title="Silhouette Coefficient",
        )
        plt.tight_layout()
        plt.show()

        # 转换为 NumPy 数组用于计算
        silhouette_scores = np.array(silhouette_scores)
        costs = np.array(costs)

        # 轮廓系数法: 选择轮廓系数最高的K值
        k_silhouette = self.K_range[np.argmax(silhouette_scores)]
        max_silhouette = max(silhouette_scores)

        # 肘部法则: 通过二阶差分找到成本函数的拐点
        diffs = np.diff(costs)
        second_diffs = np.diff(diffs)
        k_elbow = self.K_range[
            np.argmax(second_diffs) + 1
        ]  # diff 差分操作会使数组长度变短, 所以 +1

        # 综合评分法: 结合轮廓系数和成本函数 (轮廓系数权重 70%)
        silhouette_norm = (silhouette_scores - min(silhouette_scores)) / (
            max(silhouette_scores) - min(silhouette_scores)
        )
        cost_norm = (costs - min(costs)) / (max(costs) - min(costs))
        combined_scores = 0.7 * silhouette_norm + 0.3 * (1 - cost_norm)
        k_combined = self.K_range[np.argmax(combined_scores)]
        max_combined = max(combined_scores)

        print("\n" + "=" * 50)
        print(f"轮廓系数法最佳K值: {k_silhouette} (轮廓系数: {max_silhouette:.4f})")
        print(f"肘部法则最佳K值: {k_elbow} (成本: {costs[k_elbow-2]:.4f})")
        print(f"综合评分法最佳K值: {k_combined} (综合分数: {max_combined:.4f})")

        self.n_clusters = k_combined

    def plot_k_distance(self):
        """绘制 K 距离图并自动计算最优 eps 值"""
        neighbors = NearestNeighbors(
            n_neighbors=self.n_clusters + 1, metric="precomputed"
        )
        neighbors.fit(self.dist_mixed)  # 计算距离
        distances, indices = neighbors.kneighbors(self.dist_mixed)
        k_distances = np.sort(distances[:, -1])[
            ::-1
        ]  # 提取每个点的第 k 个最近邻距离并降序排列

        diffs = np.diff(k_distances)  # 计算差分
        window_size = 5  # 窗口大小 (数据集较小)
        smoothed = np.convolve(
            diffs, np.ones(window_size) / window_size, mode="valid"
        )  # 滑动窗口平滑

        # 找到最大差分点
        elbow_idx = np.argmax(smoothed) + window_size // 2
        optimal_eps = k_distances[elbow_idx]

        # K 距离图
        plt.figure(figsize=(8, 6))
        plt.plot(
            range(len(k_distances)), k_distances, marker="o", linestyle="-", color="b"
        )
        plt.axhline(
            y=optimal_eps,
            color="r",
            linestyle="--",
            label=f"Optimal eps ≈ {optimal_eps:.2f}",
        )  # 在图上标记出肘部位置
        plt.scatter([elbow_idx], [optimal_eps], color="red", zorder=5)  # 标记拐点
        plt.title(f"K-Distance Graph (k={self.n_clusters})")
        plt.xlabel("Points sorted by distance")
        plt.ylabel(f"Distance to {self.n_clusters}th nearest neighbor")
        plt.grid(True)
        plt.legend()
        plt.show()

        return optimal_eps

    def evaluate(self, labels, name):
        """评估函数"""
        unique_labels = set(labels)
        if len(unique_labels) < 2:  # 总簇数不足 2 个
            return name, None, len(unique_labels)
        if -1 in unique_labels:
            mask = labels != -1
            if mask.sum() == 0:  # 全部为噪声
                return name, None, 0
            n_valid_clusters = len(set(labels[mask]))
            if n_valid_clusters < 2:  # 有效点太少或只聚成了1个簇, 无法计算轮廓系数
                return name, None, n_valid_clusters
            score = metrics.silhouette_score(
                self.dist_mixed[mask][:, mask], labels[mask], metric="precomputed"
            )
            n_clusters = len(unique_labels) - 1
        else:
            score = metrics.silhouette_score(
                self.dist_mixed, labels, metric="precomputed"
            )
            n_clusters = len(unique_labels)
        return name, score, n_clusters

    def show_clusters(self):
        """执行 K-Prototypes, 层次聚类, DBSCAN, OPTICS 算法"""
        print("\n" + "=" * 50)
        print(f"聚类数: {self.n_clusters}, gamma={self.gamma}")
        eps_value = self.plot_k_distance()
        print(f"DBSCAN 最优 eps 值: {eps_value}")

        # K-Prototypes
        kproto = KPrototypes(
            n_clusters=self.n_clusters,
            init="Cao",
            verbose=0,
            gamma=self.gamma,
            random_state=0,
        )
        kproto.fit(self.data_array, categorical=[0])
        labels_kproto = kproto.labels_

        # 层次聚类
        agg = AgglomerativeClustering(
            n_clusters=self.n_clusters, metric="precomputed", linkage="average"
        )
        labels_agg = agg.fit_predict(self.dist_mixed)

        # DBSCAN
        dbscan = DBSCAN(metric="precomputed", eps=eps_value, min_samples=5)
        labels_dbscan = dbscan.fit_predict(self.dist_mixed)

        # OPTICS
        optics_sensitive = OPTICS(
            min_samples=5, metric="precomputed", xi=0.05, min_cluster_size=5
        )
        labels_optics_sensitive = optics_sensitive.fit_predict(self.dist_mixed)

        optics_balanced = OPTICS(
            min_samples=5, metric="precomputed", xi=0.1, min_cluster_size=15
        )
        labels_optics_balanced = optics_balanced.fit_predict(self.dist_mixed)

        optics_conservative = OPTICS(
            min_samples=5, metric="precomputed", xi=0.15, min_cluster_size=20
        )
        labels_optics_conservative = optics_conservative.fit_predict(self.dist_mixed)

        self.scores = [
            self.evaluate(labels_kproto, "K-Prototypes"),
            self.evaluate(labels_agg, "层次聚类"),
            self.evaluate(labels_dbscan, "DBSCAN"),
            self.evaluate(labels_optics_sensitive, "OPTICS (Sensitive)"),
            self.evaluate(labels_optics_balanced, "OPTICS (Balanced)"),
            self.evaluate(labels_optics_conservative, "OPTICS (Conservative)"),
        ]

        for name, score, n_clusters in self.scores:
            score = f"{score:.4f}" if score is not None else "N/A"
            print(f"{name}, 轮廓系数: {score}, 簇数: {n_clusters}")

        self.plot_clusters_scatter(
            labels_kproto, title="K-Prototypes 聚类结果 (Income vs Score)"
        )
        self.plot_clusters_scatter(labels_agg, title="层次聚类结果 (Income vs Score)")
        self.plot_radar_chart(labels_kproto, title="K-Prototypes 客户分群雷达图")

    def plot_clusters_scatter(self, labels, title="聚类结果"):
        """用 Income vs Score 展示聚类分布, 并标注性别"""
        df_viz = pd.DataFrame(
            {
                "Income": self.data_raw["Income"].values,
                "Score": self.data_raw["Score"].values,
                "Gender": self.data_raw["Gender"].values,
                "Cluster": labels,
            }
        )
        plt.figure(figsize=(8, 6))
        sns.scatterplot(
            data=df_viz,
            x="Income",
            y="Score",
            hue="Cluster",
            style="Gender",
            palette="viridis",
            s=60,
            alpha=0.8,
        )
        plt.title(title)
        plt.xlabel("Annual Income (标准化)")
        plt.ylabel("Spending Score (标准化)")
        plt.grid(True, linestyle="--", alpha=0.3)
        plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
        plt.tight_layout()
        plt.show()

    def plot_radar_chart(self, labels, title="Cluster Profiles"):
        """绘制雷达图展示每簇的特征 (Age, Income, Score) + 女性百分比"""
        # 合并数据
        df_feat = pd.DataFrame(
            {
                "Age": self.data_raw["Age"].values,
                "Income": self.data_raw["Income"].values,
                "Score": self.data_raw["Score"].values,
                "Gender": self.data_raw["Gender"].values,
                "Cluster": labels,
            }
        )

        # 计算每个簇的均值 (数值特征) + 女性比例
        cluster_profile = (
            df_feat.groupby("Cluster")
            .agg(
                Age=("Age", "mean"),
                Income=("Income", "mean"),
                Score=("Score", "mean"),
                Female_pct=("Gender", lambda x: (x == "Female").mean()),
            )
            .reset_index(drop=True)
        )

        # 归一化到 0~1 (基于整体列的最大最小值, 保证不同簇可比)
        norm_df = (cluster_profile - cluster_profile.min()) / (
            cluster_profile.max() - cluster_profile.min() + 1e-8
        )
        variables = ["Age", "Income", "Score", "Female%"]
        n_vars = len(variables)

        # 雷达图角度设置
        angles = [n / float(n_vars) * 2 * pi for n in range(n_vars)]
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
        colors = plt.cm.tab10(np.linspace(0, 1, len(norm_df)))
        for i, row in norm_df.iterrows():
            values = [row["Age"], row["Income"], row["Score"], row["Female_pct"]]
            values += values[:1]
            ax.plot(
                angles, values, "o-", linewidth=2, label=f"Cluster {i}", color=colors[i]
            )
            ax.fill(angles, values, alpha=0.1, color=colors[i])

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(variables, fontsize=12)
        ax.set_ylim(0, 1)
        ax.set_title(title, size=15, pad=20)
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
        plt.tight_layout()
        plt.show()


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
    task_map = {
        "0": "ClusterAnalysis80",
        "1": "ClusterAnalysis788",
        "2": "CustomerAnalysis",
        "3": "UniversityAnalysis",
    }

    print("请选择要运行的任务：")
    print("  0 - ClusterAnalysis (Points80.csv)")
    print("  1 - ClusterAnalysis (Points788.csv)")
    print("  2 - CustomerAnalysis")
    print("  3 - UniversityAnalysis")

    choice = input("请输入编号: ").strip()

    if choice not in task_map:
        print(f"无效输入: {choice}")
        raise SystemExit(1)

    task = task_map[choice]

    if task == "ClusterAnalysis80":
        ca = ClusterAnalysis("data/Points80.csv")
        ca.show_scatter()
        ca.elbow_method()
        ca.silhouette_coefficient()
        ca.show_KMeans()
        ca.show_GMM()
        ca.plot_k_distance()
        ca.show_DBSCAN()

    elif task == "ClusterAnalysis788":
        ca = ClusterAnalysis("data/Points788.csv")
        ca.show_scatter()
        ca.elbow_method()
        ca.silhouette_coefficient()
        ca.show_KMeans()
        ca.show_GMM()
        ca.plot_k_distance()
        ca.show_DBSCAN()

    elif task == "CustomerAnalysis":
        anlys = CustomerAnalysis("data/MallCustomers.csv")
        anlys.data_preview()  # 数据预览
        anlys.data_preprocessing()  # 数据预处理
        anlys.find_optimal_k()  # 确定最优 K 值
        anlys.show_clusters()  # 聚类结果展示

    elif task == "UniversityAnalysis":
        analysis = UniversityAnalysis("data/University.csv")
        analysis.data_preview()
        analysis.data_preprocessing()
        analysis.find_optimal_k()
        analysis.show_clusters()
