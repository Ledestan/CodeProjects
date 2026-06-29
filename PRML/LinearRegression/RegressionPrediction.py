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
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold, train_test_split

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False


class RegressionPrediction:
    def __init__(self, path: str):
        self.data = pd.read_csv(path, sep=",", header=0)
        self.X_raw = None  # 原始特征矩阵 (DataFrame)
        self.y_raw = None  # 原始目标变量 (Series)
        self.X_norm = None  # 归一化后的特征矩阵 (不含偏置列)
        self.X_train = None  # 训练集特征 (含偏置列)
        self.X_test = None  # 测试集特征 (含偏置列)
        self.y_train = None  # 训练集目标值
        self.y_test = None  # 测试集目标值
        self.theta = None  # 模型参数 (包含偏置项)
        self.loss_history = []  # 每次迭代的损失值记录
        self.scaler_min = None  # 归一化时每列的最小值
        self.scaler_max = None  # 归一化时每列的最大值

    def explore_data(self):
        """数据探索"""
        print("\n" + "=" * 50)
        print("数据基本信息: ")
        self.data.info()
        print("\n" + "=" * 50)
        print(f"数据形状: {self.data.shape}")
        print(f"描述性统计:\n{self.data.describe()}")
        print("\n" + "=" * 50)

        # 绘制所有特征的并列盒图 (不包含目标变量)
        self.data.iloc[:, :-1].boxplot(figsize=(12, 6))
        plt.title("Boxplots of All Features")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

        # 绘制所有特征与目标变量的相关系数热力图
        corr = self.data.corr()
        mask = np.tril(np.ones_like(corr, dtype=bool))  # 下三角掩码
        fig, ax = plt.subplots(figsize=(8, 6))
        # 绘制下三角热力图
        im = ax.imshow(corr.where(mask), cmap="viridis", interpolation="nearest")
        plt.colorbar(im)
        ax.set_title("Correlation Heatmap (Lower Triangle)")
        ax.set_xticks(range(len(corr.columns)))
        ax.set_xticklabels(corr.columns, rotation=90)
        ax.set_yticks(range(len(corr.columns)))
        ax.set_yticklabels(corr.columns)

        # 在下三角每个格子中添加数值文本
        for i in range(len(corr.columns)):
            for j in range(len(corr.columns)):
                if i >= j:  # 只针对下三角 (包括对角线) 显示数字
                    value = corr.iloc[i, j]
                    # 根据背景色亮度决定文字颜色 (提高可读性)
                    color = "white" if abs(value) > 0.5 else "black"
                    ax.text(
                        j,
                        i,
                        f"{value:.2f}",
                        ha="center",
                        va="center",
                        color=color,
                        fontsize=8,
                    )

        plt.tight_layout()
        plt.show()

    def min_max_normalize(self, X):
        """归一化处理"""
        X = np.array(X, dtype=float)
        self.scaler_min = X.min(axis=0)
        self.scaler_max = X.max(axis=0)
        range_ = self.scaler_max - self.scaler_min
        range_[range_ == 0] = 1

        # 返回归一化后的矩阵
        return (X - self.scaler_min) / range_

    def add_bias_column(self, X):
        """添加一列全1 (偏置项)"""
        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return np.c_[np.ones(X.shape[0]), X]

    def prepare_data(self):
        """数据准备,分离、归一化、加偏置、划分训练测试集"""
        # 分离特征与目标(最后一列为目标)
        self.X_raw = self.data.iloc[:, :-1]
        self.y_raw = self.data.iloc[:, -1]

        self.X_norm = self.min_max_normalize(self.X_raw)  # 归一化处理
        X_with_bias = self.add_bias_column(self.X_norm)  # 添加偏置列

        # 划分数据集
        X_train, X_test, y_train, y_test = train_test_split(
            X_with_bias, self.y_raw.values, test_size=0.3, random_state=0
        )
        self.X_train, self.X_test = X_train, X_test
        self.y_train, self.y_test = y_train, y_test

    def batch_gradient_descent(self, alpha, n_iter):
        """批量梯度下降"""
        X = self.X_train
        y = self.y_train
        n_features = X.shape[1]
        theta = np.zeros(n_features)
        loss_history = []
        for _ in range(n_iter):
            y_pred = X @ theta

            # 损失函数 J = 1/(2n) * Σ(y_pred - y)^2
            loss = (1 / (2 * len(y))) * np.sum((y_pred - y) ** 2)
            loss_history.append(loss)

            # 梯度 grad = (1/n) * X^T (Xθ - y)
            grad = (1 / len(y)) * (X.T @ (y_pred - y))
            theta -= alpha * grad

        return theta, loss_history

    def stochastic_gradient_descent(self, alpha, n_iter, batch_size=1):
        """随机梯度下降/小批量梯度下降"""
        X = self.X_train
        y = self.y_train
        n_samples = X.shape[0]
        n_features = X.shape[1]
        theta = np.zeros(n_features)  # 初始化参数为 0
        loss_history = []  # 记录每个 epoch 的损失

        for epoch in range(n_iter):
            # 每个 epoch 随机打乱样本顺序, 避免顺序影响
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            # 按 batch_size 遍历所有样本
            for i in range(0, n_samples, batch_size):
                X_batch = X_shuffled[i : i + batch_size]
                y_batch = y_shuffled[i : i + batch_size]
                y_pred = X_batch @ theta  # 当前 batch 的预测值
                grad = (1 / len(y_batch)) * (
                    X_batch.T @ (y_pred - y_batch)
                )  # batch 梯度
                theta -= alpha * grad  # 更新参数

            # epoch 结束后, 整个训练集上计算损失并记录
            y_pred_all = X @ theta
            loss = (1 / (2 * len(y))) * np.sum((y_pred_all - y) ** 2)
            loss_history.append(loss)

        return theta, loss_history

    def train(self, method="bgd", alpha=0.01, n_iter=500):
        """统一训练接口"""
        if method == "bgd":
            # 调用批量梯度下降, 返回最优参数和损失历史
            self.theta, self.loss_history = self.batch_gradient_descent(alpha, n_iter)
        elif method == "sgd":
            # 调用随机梯度下降 (batch_size=1), 返回最优参数和损失历史
            self.theta, self.loss_history = self.stochastic_gradient_descent(
                alpha, n_iter, batch_size=1
            )
        else:
            raise ValueError("method must be 'bgd' or 'sgd'")

    def plot_loss_curve(self, loss_history, title="Loss Curve"):
        """绘制损失收敛曲线"""
        plt.figure(figsize=(8, 6))
        plt.plot(loss_history)
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.title(title)
        plt.grid(True)
        plt.show()

    def plot_predictions(self, y_true, y_pred):
        """绘制真实值与预测值散点图"""
        plt.figure(figsize=(6, 6))
        plt.scatter(y_true, y_pred, alpha=0.6)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], "r--")
        plt.xlabel("True Values")
        plt.ylabel("Predictions")
        plt.title("True vs Predicted")
        plt.grid(True)
        plt.show()

    def baseline(self):
        """基线模型: BGD 默认参数,输出 MSE/MAE,并绘图"""
        self.train(method="bgd", alpha=0.01, n_iter=500)
        y_pred = self.X_test @ self.theta  # 预测
        mse = np.mean((y_pred - self.y_test) ** 2)  # MSE
        mae = np.mean(np.abs(y_pred - self.y_test))  # MAE
        print(f"BGD MSE: {mse:.4f}, MAE: {mae:.4f}")
        self.plot_loss_curve(self.loss_history, "BGD Loss Curve")
        self.plot_predictions(self.y_test, y_pred)

    def tune_hyperparameters(self):
        """网格搜索最佳学习率和迭代次数, 并用最佳参数重新训练, 输出调优后 MSE/MAE"""
        # 待搜索的学习率和迭代次数列表
        alphas = [0.001, 0.005, 0.01, 0.05]
        iter_list = [200, 500, 1000]

        best_loss = float("inf")
        best_params = {"alpha": None, "n_iter": None}

        # 网格搜索: 遍历所有参数组合
        for alpha in alphas:
            for n_iter in iter_list:
                # 在当前参数下训练, 获得最终的损失值
                theta, loss_hist = self.batch_gradient_descent(alpha, n_iter)
                final_loss = loss_hist[-1]
                # 更新最佳参数
                if final_loss < best_loss:
                    best_loss = final_loss
                    best_params = {"alpha": alpha, "n_iter": n_iter}

        # 使用搜索到的最佳参数重新训练模型
        self.train(
            method="bgd", alpha=best_params["alpha"], n_iter=best_params["n_iter"]
        )
        # 在测试集上预测并评估
        y_pred = self.X_test @ self.theta
        mse = np.mean((y_pred - self.y_test) ** 2)
        mae = np.mean(np.abs(y_pred - self.y_test))
        print(f"调优后 MSE: {mse:.4f}, MAE: {mae:.4f}")

    def run_sgd(self):
        """运行 SGD 并输出 MSE/MAE, 绘制损失曲线"""
        # 使用 SGD 训练, 学习率 0.01, 迭代 100 轮, batch_size=1
        theta_sgd, loss_sgd = self.stochastic_gradient_descent(
            alpha=0.01, n_iter=100, batch_size=1
        )
        # 在测试集上预测
        y_pred = self.X_test @ theta_sgd
        # 计算 MSE 和 MAE
        mse = np.mean((y_pred - self.y_test) ** 2)
        mae = np.mean(np.abs(y_pred - self.y_test))
        print(f"SGD MSE: {mse:.4f}, MAE: {mae:.4f}")
        # 绘制损失曲线
        self.plot_loss_curve(loss_sgd, "SGD Loss Curve")

    def run_regularization(self, reg_type="l2", lambda_=0.1):
        """运行正则化 (默认 L2), 输出 MSE"""
        X = self.X_train
        y = self.y_train
        n_features = X.shape[1]
        theta = np.zeros(n_features)  # 初始化参数
        alpha = 0.01  # 固定学习率
        n_iter = 500  # 迭代次数
        for _ in range(n_iter):
            y_pred = X @ theta
            # 计算梯度 (基础部分)
            grad = (1 / len(y)) * (X.T @ (y_pred - y))
            if reg_type == "l2":
                # L2 正则化: 梯度增加 (lambda_/n) * theta_j
                grad[1:] += (lambda_ / len(y)) * theta[1:]
            elif reg_type == "l1":
                # L1 正则化: 梯度增加 (lambda_/n) * sign(theta_j)
                grad[1:] += (lambda_ / len(y)) * np.sign(theta[1:])
            else:
                raise ValueError("reg_type must be 'l1' or 'l2'")
            theta -= alpha * grad  # 更新参数
        # 测试集预测并计算 MSE
        y_pred_test = self.X_test @ theta
        mse = np.mean((y_pred_test - self.y_test) ** 2)
        print(f"L2 正则化 MSE: {mse:.4f}")

    def run_feature_selection(self, top_k=5):
        """选择 top_k 个相关特征, 重新训练并输出 MSE"""
        # 获取目标变量名
        target = self.y_raw.name
        # 计算每个特征与目标的相关系数绝对值, 降序排列
        corr_abs = self.X_raw.corrwith(self.y_raw).abs().sort_values(ascending=False)
        # 选取前 top_k 个特征
        selected = corr_abs.head(top_k).index
        # 提取选中的特征矩阵 (numpy数组)
        X_sel = self.X_raw[selected].values
        # 对选中特征进行归一化
        X_sel_norm = self.min_max_normalize(X_sel)
        # 添加偏置列
        X_sel_bias = self.add_bias_column(X_sel_norm)
        # 划分训练集和测试集
        X_train_sel, X_test_sel, y_train_sel, y_test_sel = train_test_split(
            X_sel_bias, self.y_raw.values, test_size=0.3, random_state=0
        )
        # 临时替换当前训练数据为特征选择后的数据
        old_X_train, old_y_train = self.X_train, self.y_train
        self.X_train, self.y_train = X_train_sel, y_train_sel
        # 使用 BGD 训练模型
        theta_sel, _ = self.batch_gradient_descent(alpha=0.01, n_iter=500)
        # 恢复原始训练数据
        self.X_train, self.y_train = old_X_train, old_y_train
        # 在测试集上预测并计算 MSE
        y_pred = X_test_sel @ theta_sel
        mse = np.mean((y_pred - y_test_sel) ** 2)
        print(f"特征选择 MSE: {mse:.4f}")

    def run_pca(self, n_components=5):
        """PCA 降维后重新训练并输出 MSE"""
        # PCA 降维到 n_components 维
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(self.X_raw.values)
        # 对降维后的特征进行归一化
        X_pca_norm = self.min_max_normalize(X_pca)
        # 添加偏置列
        X_pca_bias = self.add_bias_column(X_pca_norm)
        # 划分训练集和测试集
        X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(
            X_pca_bias, self.y_raw.values, test_size=0.3, random_state=0
        )
        # 临时替换训练数据
        old_X_train, old_y_train = self.X_train, self.y_train
        self.X_train, self.y_train = X_train_pca, y_train_pca
        # 使用 BGD 训练模型
        theta_pca, _ = self.batch_gradient_descent(alpha=0.01, n_iter=500)
        # 恢复原始训练数据
        self.X_train, self.y_train = old_X_train, old_y_train
        # 预测并计算 MSE
        y_pred = X_test_pca @ theta_pca
        mse = np.mean((y_pred - y_test_pca) ** 2)
        print(f"PCA MSE: {mse:.4f}")

    def run_cross_validation(self, k=5, alpha=0.01, n_iter=500):
        """k折交叉验证, 输出平均 MSE"""
        # 准备完整数据集: 已归一化 + 偏置列
        X_full = self.add_bias_column(self.X_norm)
        y_full = self.y_raw.values
        # 创建 KFold 对象
        kf = KFold(n_splits=k, shuffle=True, random_state=42)
        mse_list = []
        for train_idx, val_idx in kf.split(X_full):
            # 划分当前折的训练集和验证集
            X_train_cv, X_val_cv = X_full[train_idx], X_full[val_idx]
            y_train_cv, y_val_cv = y_full[train_idx], y_full[val_idx]
            # 临时替换训练数据
            old_X_train, old_y_train = self.X_train, self.y_train
            self.X_train, self.y_train = X_train_cv, y_train_cv
            # 训练模型
            theta_cv, _ = self.batch_gradient_descent(alpha, n_iter)
            # 恢复原始训练数据
            self.X_train, self.y_train = old_X_train, old_y_train
            # 在验证集上预测并计算 MSE
            y_pred_cv = X_val_cv @ theta_cv
            mse = np.mean((y_pred_cv - y_val_cv) ** 2)
            mse_list.append(mse)
        # 输出平均 MSE
        avg_mse = np.mean(mse_list)
        print(f"5折CV平均MSE: {avg_mse:.4f}")


if __name__ == "__main__":
    reg = RegressionPrediction("data/boston.csv")
    reg.explore_data()  # 数据探索与可视化
    reg.prepare_data()  # 数据预处理
    reg.baseline()  # 基线 BGD
    reg.tune_hyperparameters()  # 超参数调优
    reg.run_sgd()  # SGD 对比
    reg.run_regularization()  # L2 正则化
    reg.run_feature_selection()  # 特征选择
    reg.run_pca()  # PCA 降维
    reg.run_cross_validation()  # 交叉验证
