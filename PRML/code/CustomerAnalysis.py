"""
客户数据处理工具

创建日期：2026-04-16
需求文件：data/Mall_Customers.csv

依赖库：
pandas>=3.0.1
numpy>=2.2.6
matplotlib>=3.10.8
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

class CustomerAnalysis:
    def __init__(self, path:str):
        self.path = path
        self.data = pd.read_csv(os.path.join(path, 'Mall_Customers.csv'), encoding ='gbk', header=None, sep=',')

    def data_load(self):
        pass

    def data_preview(self):
        pass

    def data_preprocessing(self):
        pass


if __name__ == "__main__":
    path = 'path'
    anlys = CustomerAnalysis(path)
    """
    data_load：
    1.去除第一列编号数据，处理表头"Annual Income (k$)"改为Income，"Spending Score (1-100)"为Score。
    2.处理Gender数据，Male使用10表示，Female使用01表示。
    data_preview:
    1.展示数据基本信息
    2.可视化分布展示（Age,Income,Score三个分布图）
    3.特征与性别的散点图矩阵
    data_preprocessing：
    1.数据预处理
    """