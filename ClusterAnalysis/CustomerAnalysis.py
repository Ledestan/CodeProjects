"""
客户数据分析工具

创建日期: 2026-04-16
需求文件: data\Mall_Customers.csv

依赖库:
pandas>=3.0.1
numpy>=2.2.6
seaborn>=0.13.2
matplotlib>=3.10.8
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
from sklearn.cluster import DBSCAN, OPTICS, AgglomerativeClustering
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams.update({
    "axes.facecolor": (0, 0, 0, 0), # 设置背景透明
    "axes.edgecolor": "black", # 保留边框颜色
    "figure.facecolor": "white", # 画布背景为白色
    "legend.facecolor": "white", # 图例背景为白色
})


class CustomerAnalysis:
    def __init__(self, path:str):
        self.path = path
        self.data = pd.read_csv(path, sep=",")
        self.data.drop("CustomerID", axis=1, inplace=True)
        self.data.columns = ["Gender", "Age", "Income", "Score"]
        self.data_raw = self.data.copy() # 保留原始尺度数据用于可视化
        self.K_range = range(2, 11)
        self.n_clusters = None
        self.data_array = None # data array
        self.X_num = None # data 数据部分
        self.X_cat = None # data 类别部分
        self.dist_num = None # data 数据部分距离矩阵
        self.dist_cat = None # data 类别部分距离矩阵
        self.gamma = 0.5 # 性别特征权重
        self.dist_mixed = None # data 距离混合距离
        self.scores = None # 聚类结果评估

    def data_preview(self):
        print("\n" + "=" * 50)
        print(f"数据基本信息:")
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
            axes[i].boxplot(self.data[col], vert=True, patch_artist=True, boxprops=dict(facecolor="lightblue", alpha=0.7))
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
            bp = axes[i].boxplot(box_data, patch_artist=True, tick_labels=["Male", "Female"], boxprops=dict(alpha=0.7), medianprops=dict(color="black"))
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
            diag_kws={"alpha": 0.6, "linewidth": 1.5}
        )
        g.add_legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., title="Gender")
        plt.subplots_adjust(top=0.95, right=0.85)
        plt.suptitle("Feature Distributions & Relationships", fontsize=16, y=0.98)
        plt.show()

    def data_preprocessing(self):
        """数据预处理"""
        # 数据标准化
        cols = ["Age", "Income", "Score"]
        scaler = StandardScaler()
        self.data[cols] = scaler.fit_transform(self.data[cols])

        self.data_array = self.data.values # DataFrame 转 array, KPrototypes 需要 array 输入

        # 提取数值特征矩阵和类别特征向量
        self.X_num = self.data_array[:, 1:].astype(float) # 数值部分
        self.X_cat = self.data_array[:, 0] # 类别部分

        # 数值欧氏距离 + 类别汉明距离 -> 混合距离矩阵
        self.dist_num = cdist(self.X_num, self.X_num, metric="euclidean")
        self.dist_cat = (self.X_cat[:, np.newaxis] != self.X_cat[np.newaxis, :]).astype(float)
        self.dist_mixed = self.dist_num + self.gamma * self.dist_cat

    def find_optimal_k(self):
        """综合多种指标自动确定最佳 K 值"""
        # K‑Prototypes 训练
        costs, silhouette_scores = [], []
        print("\n" + "=" * 50)
        for i in self.K_range:
            model = KPrototypes(n_clusters=i, init="Cao", verbose=0,
                                gamma=self.gamma, random_state=0)
            model.fit(self.data_array, categorical=[0])
            costs.append(model.cost_)
            score = metrics.silhouette_score(self.dist_mixed, model.labels_, metric="precomputed")
            silhouette_scores.append(score)
            print(f"K={i}, 簇内平方和={model.cost_:.4f}, 轮廓系数={score:.4f}")
        
        # 肘部法则和轮廓系数图展示
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
        ax1.plot(self.K_range, costs, "o-")
        ax1.set(xticks=self.K_range, xlabel="K Value", ylabel="Cost (混合误差)", title="Elbow Method (K-Prototypes Cost)")
        ax2.plot(self.K_range, silhouette_scores, "o-")
        ax2.set(xticks=self.K_range, xlabel="K Value", ylabel="轮廓系数", title="Silhouette Coefficient")
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
        k_elbow = self.K_range[np.argmax(second_diffs) + 1]  # diff 差分操作会使数组长度变短, 所以 +1
        
        # 综合评分法: 结合轮廓系数和成本函数 (轮廓系数权重 70%)
        silhouette_norm = (silhouette_scores - min(silhouette_scores)) / (max(silhouette_scores) - min(silhouette_scores))
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
        neighbors = NearestNeighbors(n_neighbors=self.n_clusters+1, metric="precomputed")
        neighbors.fit(self.dist_mixed)  # 计算距离
        distances, indices = neighbors.kneighbors(self.dist_mixed)
        k_distances = np.sort(distances[:, -1])[::-1] # 提取每个点的第 k 个最近邻距离并降序排列
        
        diffs = np.diff(k_distances) # 计算差分
        window_size = 5  # 窗口大小 (数据集较小)
        smoothed = np.convolve(diffs, np.ones(window_size)/window_size, mode="valid") # 滑动窗口平滑

        # 找到最大差分点
        elbow_idx = np.argmax(smoothed) + window_size // 2
        optimal_eps = k_distances[elbow_idx]
        
        # K 距离图
        plt.figure(figsize=(8, 6))
        plt.plot(range(len(k_distances)), k_distances, marker="o", linestyle="-", color="b")
        plt.axhline(y=optimal_eps, color="r", linestyle="--", label=f"Optimal eps ≈ {optimal_eps:.2f}") # 在图上标记出肘部位置
        plt.scatter([elbow_idx], [optimal_eps], color="red", zorder=5) # 标记拐点
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
        if len(unique_labels) < 2: # 总簇数不足 2 个
            return name, None, len(unique_labels)
        if -1 in unique_labels:
            mask = labels != -1
            if mask.sum() == 0:  # 全部为噪声
                return name, None, 0
            n_valid_clusters = len(set(labels[mask]))
            if n_valid_clusters < 2: # 有效点太少或只聚成了1个簇, 无法计算轮廓系数
                return name, None, n_valid_clusters
            score = metrics.silhouette_score(self.dist_mixed[mask][:, mask], labels[mask], metric="precomputed")
            n_clusters = len(unique_labels) - 1
        else:
            score = metrics.silhouette_score(self.dist_mixed, labels, metric="precomputed")
            n_clusters = len(unique_labels)
        return name, score, n_clusters

    def show_clusters(self):
        """执行 K-Prototypes, 层次聚类, DBSCAN, OPTICS 算法"""
        print("\n" + "=" * 50 )
        print(f"聚类数: {self.n_clusters}, gamma={self.gamma}")
        eps_value = self.plot_k_distance()
        print(f"DBSCAN 最优 eps 值: {eps_value}")

        # K-Prototypes
        kproto = KPrototypes(n_clusters=self.n_clusters, init="Cao", verbose=0, gamma=self.gamma, random_state=0)
        kproto.fit(self.data_array, categorical=[0])
        labels_kproto = kproto.labels_

        # 层次聚类
        agg = AgglomerativeClustering(n_clusters=self.n_clusters, metric="precomputed", linkage="average")
        labels_agg = agg.fit_predict(self.dist_mixed)

        # DBSCAN
        dbscan = DBSCAN(metric="precomputed", eps=eps_value, min_samples=5)
        labels_dbscan = dbscan.fit_predict(self.dist_mixed)

        # OPTICS
        optics_sensitive = OPTICS(min_samples=5, metric="precomputed", xi=0.05, min_cluster_size=5)
        labels_optics_sensitive = optics_sensitive.fit_predict(self.dist_mixed)

        optics_balanced = OPTICS(min_samples=5, metric="precomputed", xi=0.1, min_cluster_size=15)
        labels_optics_balanced = optics_balanced.fit_predict(self.dist_mixed)

        optics_conservative = OPTICS(min_samples=5, metric="precomputed", xi=0.15, min_cluster_size=20)
        labels_optics_conservative = optics_conservative.fit_predict(self.dist_mixed)
        
        self.scores = [
            self.evaluate(labels_kproto, "K-Prototypes"),
            self.evaluate(labels_agg, "层次聚类"),
            self.evaluate(labels_dbscan, "DBSCAN"),
            self.evaluate(labels_optics_sensitive, "OPTICS (Sensitive)"),
            self.evaluate(labels_optics_balanced, "OPTICS (Balanced)"),
            self.evaluate(labels_optics_conservative, "OPTICS (Conservative)")
        ]

        for name, score, n_clusters in self.scores:
            score = f"{score:.4f}" if score is not None else "N/A"
            print(f"{name}, 轮廓系数: {score}, 簇数: {n_clusters}")
        
        self.plot_clusters_scatter(labels_kproto, title="K-Prototypes 聚类结果 (Income vs Score)")
        self.plot_clusters_scatter(labels_agg, title="层次聚类结果 (Income vs Score)")
        self.plot_radar_chart(labels_kproto, title="K-Prototypes 客户分群雷达图")

    def plot_clusters_scatter(self, labels, title="聚类结果"):
        """用 Income vs Score 展示聚类分布, 并标注性别"""
        df_viz = pd.DataFrame({
            "Income": self.data_raw["Income"].values,
            "Score":  self.data_raw["Score"].values,
            "Gender": self.data_raw["Gender"].values,
            "Cluster": labels
        })
        plt.figure(figsize=(8,6))
        sns.scatterplot(data=df_viz, x="Income", y="Score", hue="Cluster", style="Gender", palette="viridis", s=60, alpha=0.8)
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
        df_feat = pd.DataFrame({
            "Age": self.data_raw["Age"].values,
            "Income": self.data_raw["Income"].values,
            "Score": self.data_raw["Score"].values,
            "Gender": self.data_raw["Gender"].values,
            "Cluster": labels
        })

        # 计算每个簇的均值 (数值特征) + 女性比例
        cluster_profile = df_feat.groupby("Cluster").agg(
            Age=("Age", "mean"),
            Income=("Income", "mean"),
            Score=("Score", "mean"),
            Female_pct=("Gender", lambda x: (x == "Female").mean())
        ).reset_index(drop=True)

        # 归一化到 0~1 (基于整体列的最大最小值, 保证不同簇可比)
        norm_df = (cluster_profile - cluster_profile.min()) / (cluster_profile.max() - cluster_profile.min() + 1e-8)
        variables = ["Age", "Income", "Score", "Female%"]
        n_vars = len(variables)

        # 雷达图角度设置
        angles = [n / float(n_vars) * 2 * pi for n in range(n_vars)]
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(8,8), subplot_kw=dict(polar=True))
        colors = plt.cm.tab10(np.linspace(0, 1, len(norm_df)))
        for i, row in norm_df.iterrows():
            values = [row["Age"], row["Income"], row["Score"], row["Female_pct"]]
            values += values[:1]
            ax.plot(angles, values, "o-", linewidth=2, label=f"Cluster {i}", color=colors[i])
            ax.fill(angles, values, alpha=0.1, color=colors[i])

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(variables, fontsize=12)
        ax.set_ylim(0, 1)
        ax.set_title(title, size=15, pad=20)
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    anlys = CustomerAnalysis("data/Mall_Customers.csv")
    anlys.data_preview() # 数据预览
    anlys.data_preprocessing() # 数据预处理
    anlys.find_optimal_k() # 确定最优 K 值
    anlys.show_clusters() # 聚类结果展示