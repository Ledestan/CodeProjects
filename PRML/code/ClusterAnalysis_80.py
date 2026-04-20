"""
聚类分析工具

创建日期：2026-04-02
需求文件：data/points80.txt

依赖库：
pandas>=3.0.1
matplotlib>=3.10.8
scikit-learn>=1.8.0
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import metrics

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

class ClusterAnalysis:
    def __init__(self, path:str):
        self.data = pd.read_csv(path, encoding ='gbk', header=None, sep='\t')

    def show_scatter(self, title='Scatter Plot', color='b'):
        """散点图"""
        plt.scatter(self.data.iloc[:, 0], self.data.iloc[:, 1], c=color)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(title)
        plt.tight_layout()
        plt.show()

    def elbow_method(self):
        """肘方法"""
        list_inertia = []
        for i in range(2, 11):
            model = KMeans(n_clusters=i, random_state=0)
            model.fit(self.data)
            list_inertia.append(model.inertia_)
        plt.plot(range(2, 11), list_inertia, 'o-')
        plt.xlabel('K Value')
        plt.ylabel('SSE')
        plt.title('Elbow of SSE')
        plt.tight_layout()
        plt.show()

    def silhouette_coefficient(self):
        """轮廓系数"""
        list_silhouette_score = []
        print('\n' + '=' * 50 )
        for i in range(2, 11):
            model = KMeans(n_clusters=i, random_state=0)
            model.fit(self.data)
            score = metrics.silhouette_score(self.data, model.labels_)
            print(f'第 {i} 个 K 值，轮廓系数为：{score:.4f}')
            list_silhouette_score.append(score)
        plt.plot(range(2, 11), list_silhouette_score, 'o-')
        plt.xlabel('K Value')
        plt.ylabel('轮廓系数')
        plt.title('Silhouette Coefficient')
        plt.tight_layout()
        plt.show()

        # K-Means 聚类效果评估
        print('\n' + '=' * 50 )
        best_k = list_silhouette_score.index(max(list_silhouette_score)) + 2
        print(f'最佳 K 值：{best_k}')

        model = KMeans(n_clusters=best_k, random_state=0)
        model.fit(self.data)
        print('K-Meams 聚类结果：\n', model.labels_)
        print('K-Meams 聚类中心：\n', model.cluster_centers_)

        plt.scatter(model.cluster_centers_[:, 0], model.cluster_centers_[:, 1],
                    marker='*', c='r')
        self.show_scatter('K-Meams 聚类结果散点图', model.labels_)

        # 模型评估
        print(f'轮廓系数：{metrics.silhouette_score(self.data, model.labels_)}')
        print(f'戴维斯-布尔丁指数：{metrics.davies_bouldin_score(self.data, model.labels_)}')
        print(f'卡林斯基-哈拉巴斯指数：{metrics.calinski_harabasz_score(self.data, model.labels_)}')

if __name__ == "__main__":
    path = 'data/points80.txt'
    ca = ClusterAnalysis(path)
    ca.show_scatter()
    ca.elbow_method()
    ca.silhouette_coefficient()