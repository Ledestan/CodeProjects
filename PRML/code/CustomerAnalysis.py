"""
客户数据处理工具

创建日期：2026-04-16
需求文件：data/mall_customers.csv

依赖库：
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
from kmodes.kprototypes import KPrototypes
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn import metrics

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


class CustomerAnalysis:
    def __init__(self, path:str):
        self.path = path
        self.data = pd.read_csv(os.path.join(path, 'mall_customers.csv'), sep=',')

    def data_preview(self):
        print('\n' + '=' * 50)
        print(f'数据基本信息：')
        self.data.info()
        print('\n' + '=' * 50)
        print(f'描述性统计：\n{self.data.describe()}')
        print(self.data[['Gender']].describe().T)

    def show_histograms(self):
        """展示 Age, Income, Score 直方图"""
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))

        # 绘制 Age
        axes[0].hist(self.data['Age'], bins=10, edgecolor='black')
        axes[0].set_title('Age', fontsize=14)
        axes[0].set_xlabel('Age')
        axes[0].set_ylabel('Count')

        # 绘制 Income
        axes[1].hist(self.data['Income'], bins=10, edgecolor='black')
        axes[1].set_title('Income', fontsize=14)
        axes[1].set_xlabel('Income')

        # 绘制 Score
        axes[2].hist(self.data['Score'], bins=10, edgecolor='black')
        axes[2].set_title('Score', fontsize=14)
        axes[2].set_xlabel('Score')

        plt.tight_layout()
        plt.show()

    def show_scatter_matrix(self):
        """特征与性别的散点图矩阵"""
        sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)}) # 设置样式

        # 创建 pairplot
        g = sns.pairplot(
            self.data,
            vars=['Age', 'Income', 'Score'],
            hue='Gender',
            palette=['#4682B4', '#FFC0CB'],
            diag_kind='kde',
            plot_kws={'alpha': 0.6, 's': 40, 'edgecolor': 'none'}, # 去掉点的边框，更平滑
            diag_kws={'alpha': 0.6, 'linewidth': 1.5} # 对角线加粗
        )

        g.add_legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., title='Gender') # 调整图例至表外
        plt.subplots_adjust(top=0.95, right=0.85) # 调整整体边距
        plt.suptitle('Feature Distributions & Relationships', fontsize=16, y=0.98)
        plt.show()

    def data_preprocessing(self):
        """数据预处理"""
        self.data.drop('CustomerID', axis=1, inplace=True) # 删除 ID 列
        self.data.columns = ['Gender', 'Age', 'Income', 'Score'] # 列重命名

        # 数据标准化
        cols = ['Age', 'Income', 'Score']
        scaler = StandardScaler()
        self.data[cols] = scaler.fit_transform(self.data[cols])

    def find_optimal_k(self):
        """确定最优聚类数（K 值选择）"""
        plt.figure(figsize=(8, 4))
        K_range = range(2, 11)
        data_array = self.data.values # DataFrame 转 array，KPrototypes 需要 array 输入

        # 肘部法则
        plt.subplot(1, 2, 1)
        list_cost = []
        for i in K_range:
            model = KPrototypes(n_clusters=i, init='Cao', verbose=0)
            model.fit(data_array, categorical=[0])
            list_cost.append(model.cost_)
        plt.xticks(K_range)
        plt.plot(K_range, list_cost, 'o-')
        plt.xlabel('K Value')
        plt.ylabel('Cost (混合误差)')
        plt.title('Elbow Method (K-Prototypes Cost)')

        # 轮廓系数
        plt.subplot(1, 2, 2)
        list_silhouette_score = []
        print('\n' + '=' * 50 )
        gamma = 0.5
        for i in K_range:
            # 训练 K-Prototypes
            model = KPrototypes(n_clusters=i, init='Huang', n_init=5, verbose=0, gamma=gamma, random_state=0)
            model.fit(data_array, categorical=[0])
            labels = model.labels_

            # 提取数值特征矩阵和类别特征向量
            X_num = data_array[:, 1:].astype(float) # 数值部分
            X_cat = data_array[:, 0] # 类别部分

            # 计算混合距离矩阵
            dist_num = cdist(X_num, X_num, metric='euclidean') # 数值部分：欧氏距离
            dist_cat = (X_cat[:, np.newaxis] != X_cat[np.newaxis, :]).astype(float) # 类别部分：汉明距离

            # 混合距离 = 数值距离 + gamma * 汉明距离
            dist_mixed = dist_num + gamma * dist_cat

            # 使用预计算距离矩阵计算轮廓系数
            score = metrics.silhouette_score(dist_mixed, labels, metric='precomputed')
            print(f'K={i}，轮廓系数为：{score:.4f}')
            list_silhouette_score.append(score)
        plt.xticks(range(2, 11, 1))
        plt.plot(range(2, 11), list_silhouette_score, 'o-')
        plt.xlabel('K Value')
        plt.ylabel('轮廓系数')
        plt.title('Silhouette Coefficient')

        plt.tight_layout()
        plt.show()

    def a(self):
        pass


if __name__ == "__main__":
    path = 'data'
    anlys = CustomerAnalysis(path)
    # anlys.data_preview()
    # anlys.show_histograms()
    # anlys.show_scatter_matrix()
    anlys.data_preprocessing()
    anlys.find_optimal_k()