"""
项目名称: 随机森林
创建日期: 2026-05-28

需求文件: data/healthcare_noshows.csv

依赖库:
matplotlib>=3.10.9
numpy>=2.2.6
pandas>=3.0.1
scikit-learn>=1.8.0
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

np.random.seed(0)


class RandomForest:
    def __init__(self, path: str):
        self.data = pd.read_csv(path, sep=",", header=0)

        # 存放处理后的特征和目标
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.y_prob = None

        # 特征工程需要的映射
        self.neighbourhood_freq_map = None
        self.neighbourhood_rare_threshold = 10  # 低频社区合并阈值

        self.model = None # 训练好的随机森林模型

        # 评估结果
        self.metrics = {}
        self.optimal_threshold = 0.5

        # 特征名称(用于解释)
        self.feature_names = None

    def data_explore(self):
        """数据探索"""
        print("\n" + "=" * 50)
        print("数据基本信息:")
        self.data.info()
        print("\n" + "=" * 50)
        print(f"数据形状: {self.data.shape}")
        print(f"列名及类型:\n{self.data.dtypes}")
        print("\n" + "=" * 50)
        print(f"目标变量分布:\n{self.data['Showed_up'].value_counts()}")

        # 数值列统计
        num_cols = ["Age", "Date.diff"]
        print("\n" + "=" * 50)
        print(f"数值列描述统计:\n{self.data[num_cols].describe()}")

        # 异常值检查
        print("\n" + "=" * 50)
        print(f"年龄异常: {((self.data['Age'] < 0) | (self.data['Age'] > 100)).sum()}")
        print(f"Date.diff 异常: {(self.data['Date.diff'] < 0).sum()}")

        # 类别列频次
        cat_cols = [
            "Gender",
            "Neighbourhood",
            "Scholarship",
            "Hipertension",
            "Diabetes",
            "Alcoholism",
            "Handcap",
            "SMS_received",
        ]
        for col in cat_cols:
            print(f"\n{col} 分布:\n{self.data[col].value_counts()}")

    def clean_data(self):
        """数据清洗"""
        # 删除 ID 列
        self.data.drop("PatientId", axis=1, inplace=True)
        self.data.drop("AppointmentID", axis=1, inplace=True)

        # 处理日期: 提取星期几信息后删除原始日期列
        for col in ["ScheduledDay", "AppointmentDay"]:
            self.data[col] = pd.to_datetime(self.data[col], errors="coerce")
            # 提取星期几(0 = 周一, 6 =周日)
            self.data[f"{col}_weekday"] = self.data[col].dt.dayofweek
            # 提取是否为周末
            self.data[f"{col}_is_weekend"] = (self.data[col].dt.dayofweek >= 5).astype(
                int
            )
            # 删除原始日期列
            self.data.drop(col, axis=1, inplace=True)

        # 处理异常年龄
        valid_age = (self.data["Age"] >= 0) & (self.data["Age"] <= 100)
        self.data = self.data[valid_age]

        # 处理负的 Date.diff (预约晚于就诊)
        self.data = self.data[self.data["Date.diff"] >= 0]

        # 目标变量映射
        self.data["Showed_up"] = (
            self.data["Showed_up"].astype(str).str.upper().str.strip()
        )
        self.data["Showed_up"] = self.data["Showed_up"].map({"FALSE": 0, "TRUE": 1})

    def split_data(self):
        """分层划分训练集和测试集"""
        X = self.data.drop("Showed_up", axis=1)
        y = self.data["Showed_up"]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.3, stratify=y, random_state=0
        )

        print("\n" + "=" * 50)
        print(f"随机森林 - 训练集大小: {self.X_train.shape}, 测试集大小: {self.X_test.shape}")
        print(f"训练集失约率: {self.y_train.mean():.2%}")
        print(f"测试集失约率: {self.y_test.mean():.2%}")

    def _feature_engineering(self, X, fit):
        """
        特征工程核心函数
        - 如果 fit=True, 则基于 X 拟合编码器(例如社区频率映射)
        - 否则使用已保存的映射进行转换
        """
        df = X.copy()

        # 类别编码: Gender
        df["Gender"] = df["Gender"].map({"F": 0, "M": 1})

        # 处理 Neighbourhood (高基数类别)
        if fit:
            # 计算每个社区在训练集中的出现次数
            freq = df["Neighbourhood"].value_counts()
            # 低频合并为 'rare'
            rare = freq[freq < self.neighbourhood_rare_threshold].index
            df["Neighbourhood"] = df["Neighbourhood"].apply(
                lambda x: "rare" if x in rare else x
            )
            # 构建频率映射
            self.neighbourhood_freq_map = df["Neighbourhood"].value_counts().to_dict()
            # 频率编码
            df["Neighbourhood_encoded"] = df["Neighbourhood"].map(
                self.neighbourhood_freq_map
            )
        else:
            # 先低频合并, 再映射, 如果不在映射中, 则频率设为 0
            df["Neighbourhood"] = df["Neighbourhood"].apply(
                lambda x: x if x in self.neighbourhood_freq_map else "rare"
            )
            df["Neighbourhood_encoded"] = (
                df["Neighbourhood"].map(self.neighbourhood_freq_map).fillna(0)
            )

        # 删除原始 Neighbourhood 列
        df.drop("Neighbourhood", axis=1, inplace=True)

        # Handcap 二值化 (有残障 = 1)
        df["is_handicap"] = (df["Handcap"] > 0).astype(int)
        df.drop("Handcap", axis=1, inplace=True)

        # 派生特征: 慢性病负担指数
        df["chronic_burden"] = df["Hipertension"] + df["Diabetes"] + df["Alcoholism"]

        # 年龄分组
        bins = [0, 18, 35, 60, 120]
        labels = ["child", "youth", "adult", "senior"]
        df["age_group"] = pd.cut(df["Age"], bins=bins, labels=labels, right=False)
        # 数值映射
        age_map = {"child": 0, "youth": 1, "adult": 2, "senior": 3}
        df["age_group_code"] = df["age_group"].map(age_map)
        df.drop("age_group", axis=1, inplace=True)

        # 交互特征:Date.diff * SMS_received
        df["diff_sms_interact"] = df["Date.diff"] * df["SMS_received"]

        return df

    def train(self):
        """训练随机森林模型, 包含超参数随机搜索"""
        # 在训练集上拟合特征工程, 并更新训练集和测试集
        # 对训练集进行特征工程 (fit=True)
        self.X_train = self._feature_engineering(self.X_train, fit=True)
        # 对测试集进行特征工程 (fit=False, 使用已有的映射)
        self.X_test = self._feature_engineering(self.X_test, fit=False)

        # 保存特征名称
        self.feature_names = self.X_train.columns.tolist()
        print("\n" + "=" * 50)
        print(f"特征工程完成, 共 {len(self.feature_names)} 个特征:")
        print(self.feature_names)

        # 处理类别不平衡: 使用 class_weight='balanced'
        # 计算样本权重备用
        classes = np.unique(self.y_train)
        class_weights = compute_class_weight(
            "balanced", classes=classes, y=self.y_train
        )
        class_weight_dict = dict(zip(classes, class_weights))

        # 定义随机森林基础模型
        rf_base = RandomForestClassifier(
            random_state=0, class_weight=class_weight_dict, n_jobs=-1
        )

        # 超参数网格搜索(验证固定)
        param_grid = {
            "n_estimators": [300],
            "max_depth": [12],
            "min_samples_split": [10],
            "min_samples_leaf": [3],
            "max_features": [0.3],
        }

        # 网格搜索 + 五折交叉验证
        random_search = GridSearchCV(
            estimator=rf_base,
            param_grid=param_grid,
            cv=5,
            scoring="roc_auc",
            n_jobs=-1,
            verbose=1,
        )

        random_search.fit(self.X_train, self.y_train)

        self.model = random_search.best_estimator_
        print(f"最佳交叉验证 AUC: {random_search.best_score_:.4f}")

    def evaluate(self):
        """
        在测试集上评估模型, 计算多项指标, 并可选绘制 ROC 和 PR 曲线
        同时根据业务需求(召回率优先), 寻找最优阈值
        """
        # 预测概率和类别
        self.y_prob = self.model.predict_proba(self.X_test)[:, 1]

        # 默认阈值 0.5 时的预测
        y_pred_default = (self.y_prob >= 0.5).astype(int)

        # 计算指标(针对失约类, 即 class 0)
        auc = roc_auc_score(self.y_test, self.y_prob)
        recall_lost = recall_score(self.y_test, y_pred_default)
        precision_lost = precision_score(self.y_test, y_pred_default)
        f1_lost = f1_score(self.y_test, y_pred_default)

        # 保存指标
        self.metrics = {
            "auc": auc,
            "recall_lost": recall_lost,
            "precision_lost": precision_lost,
            "f1_lost": f1_lost,
            "default_threshold": 0.5,
        }

        print("\n" + "=" * 50)
        print(f"AUC: {auc:.4f}")
        print(f"失约类召回率: {recall_lost:.4f}")
        print(f"失约类精确率: {precision_lost:.4f}")
        print(f"失约类 F1-score: {f1_lost:.4f}")
        print("\n分类报告:")
        print(
            classification_report(
                self.y_test, y_pred_default,
                labels=[0, 1],
                target_names=["失约", "如约"],
                zero_division=0
            )
        )

        # 混淆矩阵
        cm = confusion_matrix(self.y_test, y_pred_default)
        print("混淆矩阵:")
        print(cm)

        # 寻找最佳阈值 (最大化召回率的同时保证精确率不低于 0.3)
        precisions, recalls, thresholds = precision_recall_curve(
            self.y_test, self.y_prob
        )
        best_thr = 0.5
        best_recall = recall_lost
        best_precision = precision_lost
        for i, thr in enumerate(thresholds):
            if precisions[i] >= 0.3:  # 可调整
                if recalls[i] > best_recall:
                    best_recall = recalls[i]
                    best_thr = thr
                    best_precision = precisions[i]

        self.optimal_threshold = best_thr
        print(f"\n业务导向最优阈值: {best_thr:.4f}")
        print(f"对应失约类召回率: {best_recall:.4f}, 精确率: {best_precision:.4f}")

        # 使用最优阈值重新预测
        y_pred_opt = (self.y_prob >= best_thr).astype(int)
        recall_opt = recall_score(self.y_test, y_pred_opt)
        precision_opt = precision_score(self.y_test, y_pred_opt)
        self.metrics.update(
            {
                "optimal_threshold": best_thr,
                "opt_recall_lost": recall_opt,
                "opt_precision_lost": precision_opt,
            }
        )

    def plot_curves(self):
        """绘制 ROC 曲线和 PR 曲线"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        # ROC 曲线
        fpr, tpr, _ = roc_curve(self.y_test, self.y_prob)  # 以失约类为正
        axes[0].plot(fpr, tpr, label=f'ROC (AUC={self.metrics["auc"]:.4f})')
        axes[0].plot([0, 1], [0, 1], "k--")
        axes[0].set_xlabel("False Positive Rate")
        axes[0].set_ylabel("True Positive Rate (Recall)")
        axes[0].set_title("RandomForest ROC Curve (失约类为正)")
        axes[0].legend()

        # PR 曲线
        prec, rec, _ = precision_recall_curve(self.y_test, self.y_prob)
        axes[1].plot(rec, prec, label="PR Curve")
        axes[1].set_xlabel("Recall")
        axes[1].set_ylabel("Precision")
        axes[1].set_title("RandomForest Precision-Recall Curve (失约类)")
        axes[1].axhline(y=0.3, color="r", linestyle="--", label="Min Precision (0.3)")
        axes[1].legend()

        plt.tight_layout()
        plt.show()

    def plot_feature_importance(self, top_n=15):
        """绘制特征重要性(基于随机森林的 impurity importance)"""
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1][:top_n]

        plt.figure(figsize=(10, 6))
        plt.title(f"Top {top_n} 特征重要性")
        plt.barh(range(len(indices)), importances[indices], align="center")
        plt.yticks(range(len(indices)), [self.feature_names[i] for i in indices])
        plt.gca().invert_yaxis()
        plt.xlabel("重要性")
        plt.tight_layout()
        plt.show()

        # 打印重要性列表
        print("\n" + "=" * 50)
        print("特征重要性排序:")
        for i in indices:
            print(f"{self.feature_names[i]}: {importances[i]:.4f}")


class LogisticRegressionModel:
    def __init__(self, path: str):
        self.data = pd.read_csv(path, sep=",", header=0)
        self.rf = RandomForest(path)
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.y_prob = None
        self.neighbourhood_freq_map = None  # 社区频率映射(特征工程用)
        self.neighbourhood_rare_threshold = 10  # 低频社区合并阈值
        self.model = None  # 训练好的逻辑回归模型
        self.scaler = StandardScaler()  # 特征标准化器(逻辑回归必需)
        self.metrics = {}  # 评估指标字典
        self.optimal_threshold = 0.5  # 业务最优阈值
        self.feature_names = None  # 特征名称列表

    def clean_data(self):
        """数据清洗"""
        self.rf.data = self.data
        self.rf.clean_data()
        self.data = self.rf.data

    def split_data(self):
        """分层划分训练集和测试集, 使用 stratify 保持失约率与原始数据一致"""
        X = self.data.drop("Showed_up", axis=1)
        y = self.data["Showed_up"]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.3, stratify=y, random_state=0
        )
        print("\n" + "=" * 50)
        print(
            f"逻辑回归 - 训练集大小: {self.X_train.shape}, 测试集大小: {self.X_test.shape}"
        )
        print(f"训练集失约率: {self.y_train.mean():.2%}")
        print(f"测试集失约率: {self.y_test.mean():.2%}")

    def _feature_engineering(self, X, fit):
        """特征工程"""
        df = X.copy()

        # 性别编码
        df["Gender"] = df["Gender"].map({"F": 0, "M": 1})

        # 社区频率编码(高基数类别)
        if fit:
            # 训练集: 统计频率并构建映射
            freq = df["Neighbourhood"].value_counts()
            rare = freq[freq < self.neighbourhood_rare_threshold].index
            df["Neighbourhood"] = df["Neighbourhood"].apply(
                lambda x: "rare" if x in rare else x
            )
            self.neighbourhood_freq_map = df["Neighbourhood"].value_counts().to_dict()
            df["Neighbourhood_encoded"] = df["Neighbourhood"].map(
                self.neighbourhood_freq_map
            )
        else:
            # 测试集: 使用已保存的映射
            df["Neighbourhood"] = df["Neighbourhood"].apply(
                lambda x: x if x in self.neighbourhood_freq_map else "rare"
            )
            df["Neighbourhood_encoded"] = (
                df["Neighbourhood"].map(self.neighbourhood_freq_map).fillna(0)
            )
        df.drop("Neighbourhood", axis=1, inplace=True)

        # 残障二值化
        df["is_handicap"] = (df["Handcap"] > 0).astype(int)
        df.drop("Handcap", axis=1, inplace=True)

        # 慢性病负担指数
        df["chronic_burden"] = df["Hipertension"] + df["Diabetes"] + df["Alcoholism"]

        # 年龄分组
        bins = [0, 18, 35, 60, 120]
        labels = ["child", "youth", "adult", "senior"]
        df["age_group"] = pd.cut(df["Age"], bins=bins, labels=labels, right=False)
        age_map = {"child": 0, "youth": 1, "adult": 2, "senior": 3}
        df["age_group_code"] = df["age_group"].map(age_map)
        df.drop("age_group", axis=1, inplace=True)

        # 交互特征
        df["diff_sms_interact"] = df["Date.diff"] * df["SMS_received"]

        return df

    def train(self):
        """训练逻辑回归模型"""
        # 特征工程
        self.X_train = self._feature_engineering(self.X_train, fit=True)
        self.X_test = self._feature_engineering(self.X_test, fit=False)
        self.feature_names = self.X_train.columns.tolist()
        print("\n" + "=" * 50)
        print(f"逻辑回归特征工程完成,共 {len(self.feature_names)} 个特征")
        print(self.feature_names)

        # 标准化(逻辑回归对特征尺度敏感)
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)

        # 超参数网格搜索
        param_grid = {
            "C": [100],  # 正则化强度的倒数,越小正则化越强
            "l1_ratio": [0],  # L2 正则化(岭回归)
            "solver": ["lbfgs"],  # 求解器: lbfgs 适合较小数据集
        }
        # 基础逻辑回归模型: 自动处理类别不平衡, 最大迭代次数 1000 保证收敛
        lr_base = LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=0
        )
        # 网格搜索, 5 折交叉验证, 以 AUC 为优化目标
        grid_search = GridSearchCV(
            lr_base, param_grid, cv=5, scoring="roc_auc", n_jobs=-1, verbose=1
        )
        grid_search.fit(self.X_train, self.y_train)

        self.model = grid_search.best_estimator_
        print(f"最佳交叉验证 AUC: {grid_search.best_score_:.4f}")

    def evaluate(self):
        """在测试集上评估逻辑回归模型, 输出指标、混淆矩阵, 并寻找业务最优阈值"""
        # 预测概率(正类概率, 对应标签 0: 如约)
        self.y_prob = self.model.predict_proba(self.X_test)[:, 1]
        y_pred_default = (self.y_prob >= 0.5).astype(int)

        # 计算指标(重点关注失约类)
        auc = roc_auc_score(self.y_test, self.y_prob)
        recall_lost = recall_score(self.y_test, y_pred_default)
        precision_lost = precision_score(self.y_test, y_pred_default)
        f1_lost = f1_score(self.y_test, y_pred_default)

        self.metrics = {
            "auc": auc,
            "recall_lost": recall_lost,
            "precision_lost": precision_lost,
            "f1_lost": f1_lost,
            "default_threshold": 0.5,
        }

        print("\n" + "=" * 50)
        print(f"AUC: {auc:.4f}")
        print(f"失约类召回率: {recall_lost:.4f}")
        print(f"失约类精确率: {precision_lost:.4f}")
        print(f"失约类 F1-score: {f1_lost:.4f}")
        print("\n分类报告:")
        print(
            classification_report(
                self.y_test, y_pred_default,
                labels=[0, 1],
                target_names=["失约", "如约"],
                zero_division=0
            )
        )

        # 混淆矩阵
        cm = confusion_matrix(self.y_test, y_pred_default)
        print("混淆矩阵:")
        print(cm)

        # 业务导向最优阈值: 在精确率不低于 0.3 的条件下最大化召回率
        precisions, recalls, thresholds = precision_recall_curve(
            self.y_test, self.y_prob
        )
        best_thr = 0.5
        best_recall = recall_lost
        best_precision = precision_lost
        for i, thr in enumerate(thresholds):
            if precisions[i] >= 0.3:
                if recalls[i] > best_recall:
                    best_recall = recalls[i]
                    best_thr = thr
                    best_precision = precisions[i]

        self.optimal_threshold = best_thr
        print(f"\n业务导向最优阈值: {best_thr:.4f}")
        print(f"对应失约类召回率: {best_recall:.4f}, 精确率: {best_precision:.4f}")

    def plot_curves(self):
        """绘制 ROC 曲线和 PR 曲线"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        # ROC 曲线(以失约类为正例)
        fpr, tpr, _ = roc_curve(self.y_test, self.y_prob)
        axes[0].plot(fpr, tpr, label=f'ROC (AUC={self.metrics["auc"]:.4f})')
        axes[0].plot([0, 1], [0, 1], "k--")
        axes[0].set_xlabel("False Positive Rate")
        axes[0].set_ylabel("True Positive Rate (Recall)")
        axes[0].set_title("LogisticRegression ROC Curve (失约类为正)")
        axes[0].legend()

        # PR 曲线(以失约类为正例)
        prec, rec, _ = precision_recall_curve(self.y_test, self.y_prob)
        axes[1].plot(rec, prec, label="PR Curve")
        axes[1].set_xlabel("Recall")
        axes[1].set_ylabel("Precision")
        axes[1].set_title("LogisticRegression Precision-Recall Curve (失约类)")
        axes[1].axhline(y=0.3, color="r", linestyle="--", label="Min Precision (0.3)")
        axes[1].legend()

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    path = "data/healthcare_noshows.csv"

    rf = RandomForest(path)
    rf.data_explore()  # 探索数据
    rf.clean_data()  # 数据清洗
    rf.split_data()  # 划分数据集
    rf.train()  # 划分数据集
    rf.evaluate()  # 评估模型
    rf.plot_curves()  # 绘制曲线
    rf.plot_feature_importance()  # 绘制特征重要性

    lr = LogisticRegressionModel(path)
    lr.clean_data()  # 数据清洗
    lr.split_data()  # 划分数据集
    lr.train()  # 划分数据集
    lr.evaluate()  # 划分数据集
    lr.plot_curves()  # 绘制曲线
