import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_diabetes

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False


def load_data(path: str = "data/diabetes.csv"):
    """从 Sklearn 加载糖尿病数据集，失败则回退到本地 CSV"""
    try:
        print("尝试从 Sklearn 加载数据...")
        diabetes = load_diabetes()
        df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
        df["target"] = diabetes.target
        return df
    except Exception as e1:
        print(f"从 Sklearn 加载失败: {e1}")
        try:
            print("尝试从本地文件加载...")
            # 获取本文件所在路径
            base_dir = os.path.dirname(os.path.abspath(__file__))
            file_path = os.path.join(base_dir, path)
            df = pd.read_csv(file_path)
            return df
        except Exception as e2:
            print(f"数据加载失败: {e2}")
            return None


def explore_data(df):
    """打印数据集基本信息"""
    print("\n" + "=" * 50)
    print(f"数据形状: {df.shape}")
    print("\n" + "=" * 50)
    print("数据信息: ")
    df.info()
    print("\n" + "=" * 50)
    print("数据描述: ")
    print(df.describe())


def plot_correlation(df):
    """绘制特征相关性热力图"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    df = load_data()
    if df is not None:
        explore_data(df)
        plot_correlation(df)
