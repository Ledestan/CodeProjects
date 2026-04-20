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

class CustomerAnalysis:
    def __init__(self, path:str):
        self.path = path
        self.data = pd.read_csv(os.path.join(path, 'Mall_Customers.csv'), encoding ='gbk', header=None, sep=',')

if __name__ == "__main__":
    path = 'path'
    anlys = CustomerAnalysis(path)