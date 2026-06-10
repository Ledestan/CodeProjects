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

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import (RandomizedSearchCV, StratifiedKFold,
                                     train_test_split)
from sklearn.utils.class_weight import compute_class_weight

warnings.filterwarnings("ignore")

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

        # 特征工程需要的映射
        self.neighbourhood_freq_map = None
        self.neighbourhood_rare_threshold = 10  # 低频社区合并阈值

        # 训练好的模型
        self.model = None
        self.best_params = None

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
        self.data["Showed_up"] = self.data["Showed_up"].map({"FALSE": 1, "TRUE": 0})

    def split_data(self):
        """分层划分训练集和测试集"""
        X = self.data.drop("Showed_up", axis=1)
        y = self.data["Showed_up"]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.3, stratify=y, random_state=0
        )

        print("\n" + "=" * 50)
        print(f"训练集大小: {self.X_train.shape}, 测试集大小: {self.X_test.shape}")
        print(f"训练集失约率: {1 - self.y_train.mean():.2%}")
        print(f"测试集失约率: {1 - self.y_test.mean():.2%}")

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

    def train(self, n_iter_search=20, cv_folds=5):
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

        # 超参数搜索空间
        param_dist = {
            "n_estimators": [100, 200, 300],
            "max_depth": [10, 15, 20, 25, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", "log2", 0.3, 0.5],
        }

        # 随机搜索 + 分层交叉验证
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=0)
        random_search = RandomizedSearchCV(
            rf_base,
            param_distributions=param_dist,
            n_iter=n_iter_search,
            cv=cv,
            scoring="roc_auc",
            random_state=0,
            n_jobs=-1,
            verbose=1,
        )

        random_search.fit(self.X_train, self.y_train)

        self.model = random_search.best_estimator_
        self.best_params = random_search.best_params_

        print(f"最佳参数: {self.best_params}")
        print(f"最佳交叉验证 AUC: {random_search.best_score_:.4f}")

    def evaluate(self, plot_curves=True):
        """
        在测试集上评估模型, 计算多项指标, 并可选绘制 ROC 和 PR 曲线
        同时根据业务需求(召回率优先), 寻找最优阈值
        """
        # 预测概率和类别
        y_prob = self.model.predict_proba(self.X_test)[:, 1]

        # 默认阈值 0.5 时的预测
        y_pred_default = (y_prob >= 0.5).astype(int)

        # 计算指标(针对失约类, 即 class 0)
        auc = roc_auc_score(self.y_test, y_prob)
        recall_lost = recall_score(self.y_test, y_pred_default, pos_label=0)
        precision_lost = precision_score(self.y_test, y_pred_default, pos_label=0)
        f1_lost = f1_score(self.y_test, y_pred_default, pos_label=0)

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
                self.y_test, y_pred_default, target_names=["失约", "如约"]
            )
        )

        # 混淆矩阵
        cm = confusion_matrix(self.y_test, y_pred_default)
        print("混淆矩阵:")
        print(cm)

        # 寻找最佳阈值 (最大化召回率的同时保证精确率不低于 0.3)
        precisions, recalls, thresholds = precision_recall_curve(
            self.y_test, y_prob, pos_label=0
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
        y_pred_opt = (y_prob >= best_thr).astype(int)
        recall_opt = recall_score(self.y_test, y_pred_opt, pos_label=0)
        precision_opt = precision_score(self.y_test, y_pred_opt, pos_label=0)
        self.metrics.update(
            {
                "optimal_threshold": best_thr,
                "opt_recall_lost": recall_opt,
                "opt_precision_lost": precision_opt,
            }
        )

        if plot_curves:
            self._plot_curves(y_prob)

        return self.metrics

    def _plot_curves(self, y_prob):
        """绘制 ROC 曲线和 PR 曲线"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        # ROC 曲线
        fpr, tpr, _ = roc_curve(self.y_test, y_prob, pos_label=0)  # 以失约类为正
        axes[0].plot(fpr, tpr, label=f'ROC (AUC={self.metrics["auc"]:.4f})')
        axes[0].plot([0, 1], [0, 1], "k--")
        axes[0].set_xlabel("False Positive Rate")
        axes[0].set_ylabel("True Positive Rate (Recall)")
        axes[0].set_title("ROC Curve (失约类为正)")
        axes[0].legend()

        # PR 曲线
        prec, rec, _ = precision_recall_curve(self.y_test, y_prob, pos_label=0)
        axes[1].plot(rec, prec, label="PR Curve")
        axes[1].set_xlabel("Recall")
        axes[1].set_ylabel("Precision")
        axes[1].set_title("Precision-Recall Curve (失约类)")
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


if __name__ == "__main__":
    rf = RandomForest("data/healthcare_noshows.csv")
    rf.data_explore()  # 探索数据
    rf.clean_data()  # 数据清洗
    rf.split_data()  # 划分数据集
    rf.train(n_iter_search=10, cv_folds=5)  # 训练模型
    rf.evaluate()  # 评估模型
    rf.plot_feature_importance()  # 绘制特征重要性
