"""
项目名称: 随机森林
创建日期: 2026-05-28

需求文件: data

依赖库:
matplotlib>=3.10.9
numpy>=2.2.6
pandas>=3.0.1
scikit-learn>=1.8.0
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False


class RandomForest:
    def __init__(self, path: str):
        self.data = pd.read_csv(path, sep=",", head=0)
        

if __name__ == "__main__":
    rf = RandomForest("data/healthcare_noshows.csv")
