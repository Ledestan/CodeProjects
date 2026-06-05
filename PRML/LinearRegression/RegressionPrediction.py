"""
项目名称: 回归预测
创建日期: 2026-06-05

需求文件: data/boston.csv

依赖库:
matplotlib>=3.10.9
numpy>=2.2.6
pandas>=3.0.1
scikit-learn>=1.8.0
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False


class RegressionPrediction:
    def __init__(self, path: str):
        self.data = pd.read_csv(
            path, sep=",", header=None
        )


if __name__ == "__main__":
    Reg = RegressionPrediction("data/boston.csv")