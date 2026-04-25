"""
客户数据处理工具

创建日期: 2026-04-16
需求文件: data/Mall_Customers.csv

依赖库:
pandas>=3.0.1
numpy>=2.2.6
seaborn>=0.13.2
matplotlib>=3.10.8
kmodes>=0.12.2
scipy>=1.17.1
scikit-learn>=1.8.0
"""

import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from math import pi
from kmodes.kprototypes import KPrototypes
from scipy.spatial.distance import cdist
from sklearn import metrics
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False


class CustomerAnalysis:
    def __init__(self, path:str):
        self.path = path
        self.data = pd.read_csv(os.path.join(path, "Mall_Customers.csv"), sep=",")
        self.K_range = range(2, 11)
        self.n_clusters = None
        self.data_array = None # data array
        self.X_num = None # data 数据部分
        self.X_cat = None # data 类别部分
        self.dist_num = None # data 数据部分距离矩阵
        self.dist_cat = None # data 类别部分距离矩阵
        self.gamma = 0.5 # 性别特征权重
        self.dist_mixed = None # data 距离混合距离

    def data_preview(self):
        print("\n" + "=" * 50)
        print(f"数据基本信息:")
        self.data.info()
        print("\n" + "=" * 50)
        print(f"描述性统计:\n{self.data.describe()}")
        print(self.data[["Gender"]].describe().T)

        self.show_histograms()

    def show_histograms(self):
        """展示 Age, Income, Score 直方图"""
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))

        # 绘制 Age
        axes[0].hist(self.data["Age"], bins=10, edgecolor="black")
        axes[0].set_title("Age", fontsize=14)
        axes[0].set_xlabel("Age")
        axes[0].set_ylabel("Count")

        # 绘制 Income
        axes[1].hist(self.data["Income"], bins=10, edgecolor="black")
        axes[1].set_title("Income", fontsize=14)
        axes[1].set_xlabel("Income")

        # 绘制 Score
        axes[2].hist(self.data["Score"], bins=10, edgecolor="black")
        axes[2].set_title("Score", fontsize=14)
        axes[2].set_xlabel("Score")

        plt.tight_layout()
        plt.show()

        self.show_scatter_matrix()

    def show_scatter_matrix(self):
        """特征与性别的散点图矩阵"""
        sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)}) # 设置样式

        # 创建 pairplot
        g = sns.pairplot(
            self.data,
            vars=["Age", "Income", "Score"],
            hue="Gender",
            palette=["#4682B4", "#FFC0CB"],
            diag_kind="kde",
            plot_kws={"alpha": 0.6, "s": 40, "edgecolor": "none"}, # 去掉点的边框, 更平滑
            diag_kws={"alpha": 0.6, "linewidth": 1.5} # 对角线加粗
        )

        g.add_legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., title="Gender") # 调整图例至表外
        plt.subplots_adjust(top=0.95, right=0.85) # 调整整体边距
        plt.suptitle("Feature Distributions & Relationships", fontsize=16, y=0.98)
        plt.show()

    def data_preprocessing(self):
        """数据预处理"""
        self.data.drop("CustomerID", axis=1, inplace=True) # 删除 ID 列
        self.data.columns = ["Gender", "Age", "Income", "Score"] # 列重命名

        # 数据标准化
        cols = ["Age", "Income", "Score"]
        scaler = StandardScaler()
        self.data[cols] = scaler.fit_transform(self.data[cols])

        self.data_array = self.data.values # DataFrame 转 array, KPrototypes 需要 array 输入

        # 提取数值特征矩阵和类别特征向量
        self.X_num = self.data_array[:, 1:].astype(float) # 数值部分
        self.X_cat = self.data_array[:, 0] # 类别部分

        # 计算混合距离矩阵
        self.dist_num = cdist(self.X_num, self.X_num, metric="euclidean") # 数值部分: 欧氏距离
        self.dist_cat = (self.X_cat[:, np.newaxis] != self.X_cat[np.newaxis, :]).astype(float) # 类别部分: 汉明距离

        # 混合距离 = 数值距离 + gamma * 汉明距离
        self.dist_mixed = self.dist_num + self.gamma * self.dist_cat

    def find_optimal_k(self):
        """确定最优聚类数 (K 值选择)"""
        plt.figure(figsize=(8, 4))
        
        # 肘部法则
        plt.subplot(1, 2, 1)
        costs = []
        for i in self.K_range:
            model = KPrototypes(n_clusters=i, init="Cao", verbose=0)
            model.fit(self.data_array, categorical=[0])
            costs.append(model.cost_)
        plt.xticks(self.K_range)
        plt.plot(self.K_range, costs, "o-")
        plt.xlabel("K Value")
        plt.ylabel("Cost (混合误差)")
        plt.title("Elbow Method (K-Prototypes Cost)")

        # 轮廓系数
        plt.subplot(1, 2, 2)
        silhouette_scores = []
        print("\n" + "=" * 50 )
        for i in self.K_range:
            # 训练 K-Prototypes
            model = KPrototypes(n_clusters=i, init="Cao", verbose=0, gamma=self.gamma, random_state=0)
            model.fit(self.data_array, categorical=[0])
            labels = model.labels_

            # 使用预计算距离矩阵计算轮廓系数
            score = metrics.silhouette_score(self.dist_mixed, labels, metric="precomputed")
            print(f"K={i}, 轮廓系数为: {score:.4f}")
            silhouette_scores.append(score)
        plt.xticks(range(2, 11, 1))
        plt.plot(self.K_range, silhouette_scores, "o-")
        plt.xlabel("K Value")
        plt.ylabel("轮廓系数")
        plt.title("Silhouette Coefficient")

        plt.tight_layout()
        plt.show()

        self.determine_optimal_k(silhouette_scores, costs)

    def determine_optimal_k(self, silhouette_scores, costs):
        """综合多种指标自动确定最佳 K 值"""
        # 强制转换为 NumPy 数组计算
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

    def show_clusters(self):
        """执行 K-Prototypes 和层次聚类算法"""
        print("\n" + "=" * 50 )
        print(f"聚类数: {self.n_clusters}, gamma={self.gamma}")

        # K-Prototypes
        kproto = KPrototypes(n_clusters=self.n_clusters, init="Cao", verbose=0, gamma=self.gamma, random_state=0)
        kproto.fit(self.data_array, categorical=[0])
        labels_kproto = kproto.labels_

        # 层次聚类
        agg = AgglomerativeClustering(n_clusters=self.n_clusters, metric="precomputed", linkage="average")
        labels_agg = agg.fit_predict(self.dist_mixed)

        # 评估函数
        def evaluate(labels, name):
            unique_labels = set(labels)
            if -1 in unique_labels:
                mask = labels != -1
                score = metrics.silhouette_score(self.dist_mixed[mask][:, mask], labels[mask], metric="precomputed")
                n_clusters = len(unique_labels) - 1
            else:
                score = metrics.silhouette_score(self.dist_mixed, labels, metric="precomputed")
                n_clusters = len(unique_labels)
            print(f"{name}, 轮廓系数: {score:.4f}, 簇数: {n_clusters}")
        
        evaluate(labels_kproto, "K-Prototypes")
        evaluate(labels_agg, "层次聚类")

        self.plot_clusters_scatter(labels_kproto, title="K-Prototypes 聚类结果 (Income vs Score)")
        self.plot_clusters_scatter(labels_agg, title="层次聚类结果 (Income vs Score)")
        self.plot_radar_chart(labels_kproto, title="K-Prototypes 客户分群雷达图")

    def plot_clusters_scatter(self, labels, title="聚类结果"):
        """用 Income vs Score 展示聚类分布, 并标注性别"""
        df_viz = pd.DataFrame({
            "Income": self.data["Income"].values,
            "Score":  self.data["Score"].values,
            "Gender": self.data["Gender"].values,
            "Cluster": labels
        })
        plt.figure(figsize=(8,6))
        sns.scatterplot(data=df_viz, x="Income", y="Score", hue="Cluster", style="Gender",
                        palette="viridis", s=60, alpha=0.8)
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
            "Age": self.data["Age"].values,
            "Income": self.data["Income"].values,
            "Score": self.data["Score"].values,
            "Gender": self.data["Gender"].values,
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
    path = "data"
    anlys = CustomerAnalysis(path)
    # anlys.data_preview() # 数据预览
    anlys.data_preprocessing()
    anlys.find_optimal_k()
    anlys.show_clusters()