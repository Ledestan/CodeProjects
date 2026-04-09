import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

class ClusterAnalysis:
    def __init__(self, path:str):
        self.path = path
        self.data = pd.read_csv(os.path.join(path, 'points788.txt'), encoding ='gbk', header=None, sep=',')
    
    def show_scatter(self, title='Scatter Plot', color='b'):
        """散点图"""
        plt.scatter(self.data.iloc[:, 0], self.data.iloc[:, 1], s=10, c=color)
        # plt.axis('off') # 隐藏坐标轴
        plt.xticks([]) # 隐藏 x 轴数据
        plt.yticks([]) # 隐藏 y 轴数据
        plt.title(title)
        plt.tight_layout()
        # plt.show()

    def show_KMeans(self):
        model = KMeans(n_clusters=7, random_state=28)
        model.fit(self.data)
        self.show_scatter('K-Meams 聚类结果散点图', color=model.labels_)
        plt.scatter(model.cluster_centers_[:, 0], model.cluster_centers_[:, 1], marker='*', s=50, c='r')
        # plt.show()

    def show_GMM(self):
        model = GaussianMixture(n_components=7, covariance_type='full', random_state=28)
        model.fit(self.data)
        self.show_scatter('GMM 聚类结果散点图', color=model.predict(self.data))
        plt.scatter(model.means_[:, 0], model.means_[:, 1], marker='*', s=50, c='r')
        # plt.show()

if __name__ == "__main__":
    path = 'data'
    anlys = ClusterAnalysis(path)

    plt.figure(figsize=(9, 3))
    plt.subplot(1, 3, 1)
    anlys.show_scatter()

    plt.subplot(1, 3, 2)
    anlys.show_KMeans()

    plt.subplot(1, 3, 3)
    anlys.show_GMM()

    plt.tight_layout()
    plt.show()