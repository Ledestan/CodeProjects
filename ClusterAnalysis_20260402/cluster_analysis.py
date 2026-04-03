import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

class ClusterAnalysis:
    def __init__(self, path:str):
        self.path = path
        self.df = pd.read_csv(os.path.join(path, 'points80.txt'), encoding ='gbk', header=None, sep='\t')

    def show_scatter(self, color=None):
        """散点图"""
        plt.scatter(self.df.iloc[:, 0], self.df.iloc[:, 1], color)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Scatter Plot')
        plt.tight_layout()
        plt.show()
    
    def elbow_method(self):
        """肘部法则"""
        list_inertia = []
        for i in range(1, 11):
            model = KMeans(n_clusters=i, random_state=0)
            model.fit(self.df)
            list_inertia.append(model.inertia_)
        plt.plot(range(1, 11), list_inertia, 'o-')
        plt.xlabel('K Value')
        plt.ylabel('SSE')
        plt.title('Elbow Method')
        plt.tight_layout()
        plt.show()
    
if __name__ == "__main__":
    path = 'ClusterAnalysis_20260402/data'
    anlys = ClusterAnalysis(path)
    # anlys.show_scatter()
    anlys.elbow_method()