"""
项目名称: 决策树
创建日期: 2026-05-21

需求文件: data

依赖库:
matplotlib>=3.10.9
numpy>=2.2.6
pandas>=3.0.1
scikit-learn>=1.8.0
graphviz>=0.21

代码结构总览:
DecisionNode                ← 定义树节点
DecisionTree                ← CART 决策树主类
  ├── __init__              ← 加载 CSV 文件
  ├── data_processing       ← 独热编码 + 数值化
  ├── _gini                 ← 基尼指数
  ├── _gini_gain            ← 基尼增益
  ├── _best_split           ← 寻找最优分裂点
  ├── _build_tree           ← 递归建树
  ├── fit / predict / score ← 训练、预测、评估
  ├── export_graphviz       ← 生成 Graphviz 对象
  └── plot_tree_in_memory   ← 内存渲染 + matplotlib 显示
main                        ← 训练、对比、绘图
"""

import io
from collections import Counter

import graphviz
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz as sk_export_graphviz

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False


class DecisionNode:
    """决策树节点"""

    def __init__(
        self, feature_index=None, threshold=None, left=None, right=None, *, value=None
    ):
        """
        初始化节点

        Parameters
        ----------
        feature_index : int, optional
            分裂特征在特征矩阵中的列索引, 叶节点为 None
        threshold : float, optional
            分裂阈值, 叶节点为 None
        left : DecisionNode, optional
            左子树 (满足 <= threshold)
        right : DecisionNode, optional
            右子树
        value : int, optional
            叶节点类别 (多数类), 内部节点为 None
        """
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        self.samples = 0  # 节点样本数
        self.gini = 0.0  # 节点基尼指数


class DecisionTree:
    """西瓜数据集 CART 决策树"""

    def __init__(self, path: str):
        """
        Parameters
        ----------
        path : str
            CSV 文件路径, 要求包含列名: 编号, 色泽, 根蒂, 敲声, 纹理, 脐部, 触感, 密度, 含糖率, 好瓜
        """
        self.data = pd.read_csv(path, sep=",", encoding="gbk", header=0)
        self.X = None
        self.y = None
        self.feature_names = None
        self.root = None

    def data_processing(self):
        """
        数据预处理: 删除编号列, 标签映射为 0/1, 离散特征独热编码, 连续特征保留原值.
        处理后 self.X 为 float 型 numpy 数组, self.y 为 int 型向量.
        """
        self.data.drop("编号", axis=1, inplace=True)  # 删除编号列
        self.data["好瓜"] = self.data["好瓜"].map({"是": 1, "否": 0})  # 标签映射

        # 分离特征和标签
        y = self.data["好瓜"]
        X_df = self.data.drop("好瓜", axis=1)

        categorical_cols = [
            "色泽",
            "根蒂",
            "敲声",
            "纹理",
            "脐部",
            "触感",
        ]  # 指定离散特征列名
        existing_cat_cols = [
            col for col in categorical_cols if col in X_df.columns
        ]  # 检查列是否存在
        numerical_cols = [
            col for col in X_df.columns if col not in existing_cat_cols
        ]  # 连续特征列为剩下的列

        X_encoded = pd.get_dummies(
            X_df, columns=existing_cat_cols, drop_first=False
        )  # 独热编码
        X_encoded = X_encoded.astype(float)  # 转为数值类型

        # 转为 numpy 数组
        self.X = X_encoded.values
        self.y = y.values.astype(int)
        self.feature_names = list(X_encoded.columns)

        print("\n" + "=" * 50)
        print(f"特征维度 {self.X.shape}, 标签维度 {self.y.shape}")
        print(f"特征名: \n{self.feature_names}")

    # ==================== CART 核心算法 ====================
    def _gini(self, y: np.ndarray) -> float:
        """
        计算基尼指数 Gini(D) = 1 - Σ p_k^2

        Parameters
        ----------
        y : np.ndarray
            标签数组

        Returns
        -------
        float
            基尼指数, 越小纯度越高
        """
        _, counts = np.unique(y, return_counts=True)
        probs = counts / len(y)
        return 1 - np.sum(probs**2)  # 越小纯度越高

    def _gini_gain(
        self, parent_y: np.ndarray, left_y: np.ndarray, right_y: np.ndarray
    ) -> float:
        """
        计算分裂后的基尼增益

        Parameters
        ----------
        parent_y : np.ndarray
            父节点标签
        left_y : np.ndarray
            左子节点标签
        right_y : np.ndarray
            右子节点标签

        Returns
        -------
        float
            基尼增益 = 父节点基尼 - 加权左右子节点基尼
        """
        n = len(parent_y)
        n_left, n_right = len(left_y), len(right_y)
        if n_left == 0 or n_right == 0:
            return 0.0
        gini_parent = self._gini(parent_y)
        gini_split = (n_left / n) * self._gini(left_y) + (n_right / n) * self._gini(
            right_y
        )
        return gini_parent - gini_split

    def _best_split(self, X: np.ndarray, y: np.ndarray):
        """
        遍历所有特征和候选阈值, 找到使基尼增益最大的分裂点

        Parameters
        ----------
        X : np.ndarray
            特征矩阵
        y : np.ndarray
            标签向量

        Returns
        -------
        tuple (feature_index, threshold, gain)
            若无法分裂则返回 (None, None, 0.0)
        """
        n_samples, n_features = X.shape
        if n_samples <= 1:
            return None, None, 0.0

        best_gain = -1.0
        best_feature = None
        best_threshold = None

        for f in range(n_features):
            feature_vals = X[:, f]
            unique_vals = np.unique(feature_vals)
            if len(unique_vals) <= 1:
                continue
            # 候选阈值: 相邻值的中点
            thresholds = (unique_vals[:-1] + unique_vals[1:]) / 2.0
            for th in thresholds:
                left_mask = feature_vals <= th
                left_y = y[left_mask]
                right_y = y[~left_mask]
                if len(left_y) == 0 or len(right_y) == 0:
                    continue
                gain = self._gini_gain(y, left_y, right_y)
                if gain > best_gain:
                    best_gain = gain
                    best_feature = f
                    best_threshold = th
        return best_feature, best_threshold, best_gain

    def _build_tree(
        self,
        X: np.ndarray,
        y: np.ndarray,
        depth: int,
        max_depth: int,
        min_samples_split: int,
    ):
        """
        递归构建 CART 决策树

        Parameters
        ----------
        X : np.ndarray
            当前节点特征矩阵
        y : np.ndarray
            当前节点标签
        depth : int
            当前深度 (根节点为 0)
        max_depth : int or None
            最大深度限制, None 表示不限制
        min_samples_split : int
            节点再分裂所需的最小样本数

        Returns
        -------
        DecisionNode
            当前子树的根节点
        """
        n_samples = len(y)

        # 终止条件: 纯度100%
        if len(np.unique(y)) == 1:
            leaf = DecisionNode(value=y[0])
            leaf.samples = n_samples
            leaf.gini = self._gini(y)
            return leaf

        # 终止条件: 达到最大深度或样本数过少
        if (
            max_depth is not None and depth >= max_depth
        ) or n_samples < min_samples_split:
            majority = Counter(y).most_common(1)[0][0]
            leaf = DecisionNode(value=majority)
            leaf.samples = n_samples
            leaf.gini = self._gini(y)
            return leaf

        feature, threshold, gain = self._best_split(X, y)
        # 终止条件: 无法再分裂
        if feature is None or gain <= 0.0:
            majority = Counter(y).most_common(1)[0][0]
            leaf = DecisionNode(value=majority)
            leaf.samples = n_samples
            leaf.gini = self._gini(y)
            return leaf

        # 分裂数据
        left_mask = X[:, feature] <= threshold
        X_left, y_left = X[left_mask], y[left_mask]
        X_right, y_right = X[~left_mask], y[~left_mask]

        # 递归构建子树
        left_child = self._build_tree(
            X_left, y_left, depth + 1, max_depth, min_samples_split
        )
        right_child = self._build_tree(
            X_right, y_right, depth + 1, max_depth, min_samples_split
        )

        node = DecisionNode(
            feature_index=feature,
            threshold=threshold,
            left=left_child,
            right=right_child,
        )
        node.samples = n_samples
        node.gini = self._gini(y)
        return node

    def fit(self, X: np.ndarray, y: np.ndarray, max_depth=None, min_samples_split=2):
        """
        训练决策树模型

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            训练集特征矩阵
        y : np.ndarray, shape (n_samples,)
            训练集标签向量
        max_depth : int or None, default=None
            树的最大深度, None 表示不限制
        min_samples_split : int, default=2
            内部节点再分裂所需的最小样本数
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = self._build_tree(X, y, 0, max_depth, min_samples_split)

    def _predict_single(self, x: np.ndarray, node: DecisionNode) -> int:
        """
        对单个样本递归预测类别

        Parameters
        ----------
        x : np.ndarray
            单个样本特征向量
        node : DecisionNode
            当前遍历的节点

        Returns
        -------
        int
            预测的类别标签 (0 或 1)
        """
        if node.value is not None:
            return node.value
        if x[node.feature_index] <= node.threshold:
            return self._predict_single(x, node.left)
        else:
            return self._predict_single(x, node.right)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        批量预测样本类别

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            特征矩阵

        Returns
        -------
        np.ndarray, shape (n_samples,)
            预测标签数组
        """
        if self.root is None:
            raise RuntimeError("模型尚未训练, 请先调用 fit().")
        return np.array([self._predict_single(x, self.root) for x in X])

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        计算模型准确率

        Parameters
        ----------
        X : np.ndarray
            特征矩阵
        y : np.ndarray
            真实标签

        Returns
        -------
        float
            分类准确率
        """
        pred = self.predict(X)
        return np.mean(pred == y)

    # ==================== 统计信息 ====================
    def _depth_helper(self, node):
        """
        递归计算子树的深度, 从当前节点开始计算.
        叶节点或空节点深度为 0, 内部节点深度为 1 + 左右子树最大深度.
        """
        if node is None or node.value is not None:  # 空节点或叶节点
            return 0
        return 1 + max(self._depth_helper(node.left), self._depth_helper(node.right))

    def get_depth(self):
        """返回整棵树的深度 (根节点深度为0)"""
        if self.root is None:
            return 0
        return self._depth_helper(self.root)

    def _leaves_helper(self, node):
        """递归统计子树中叶节点的个数"""
        if node is None:
            return 0
        if node.value is not None:  # 叶节点
            return 1
        return self._leaves_helper(node.left) + self._leaves_helper(node.right)

    def get_n_leaves(self):
        """返回整棵树的叶节点个数"""
        if self.root is None:
            return 0
        return self._leaves_helper(self.root)

    def _nodes_helper(self, node):
        """递归统计子树的总节点数 (包括内部节点和叶节点)"""
        if node is None:
            return 0
        return 1 + self._nodes_helper(node.left) + self._nodes_helper(node.right)

    def get_n_nodes(self):
        """返回整棵树的总节点数"""
        if self.root is None:
            return 0
        return self._nodes_helper(self.root)

    # ==================== 可视化 ====================
    def export_graphviz(self, feature_names=None, class_names=None) -> graphviz.Digraph:
        """
        将决策树导出为 graphviz.Digraph 对象, 可在内存中渲染

        Parameters
        ----------
        feature_names : list of str, optional
            特征名称列表, 默认使用 self.feature_names
        class_names : list of str, optional
            类别名称, 默认 ["否", "是"]

        Returns
        -------
        graphviz.Digraph
            决策树图对象
        """
        if self.root is None:
            raise RuntimeError("模型尚未训练.")

        if feature_names is None:
            feature_names = self.feature_names
        if class_names is None:
            class_names = ["否", "是"]  # 0 -> 否, 1 -> 是

        dot = graphviz.Digraph(comment="决策树")
        dot.attr("node", fontname="SimHei")
        dot.attr("edge", fontname="SimHei")

        def add_node(node, parent_id=None, edge_label=""):
            node_id = str(id(node))
            if node.value is not None:
                label = f"类别: {class_names[node.value]}\n样本数: {node.samples}\nGini: {node.gini:.3f}"
                dot.node(
                    node_id,
                    label=label,
                    shape="box",
                    style="filled",
                    fillcolor="lightyellow",
                )
            else:
                feat = feature_names[node.feature_index]
                label = f"{feat} <= {node.threshold:.4f}\n样本数: {node.samples}\nGini: {node.gini:.3f}"
                dot.node(
                    node_id,
                    label=label,
                    shape="box",
                    style="filled",
                    fillcolor="lightblue",
                )
            if parent_id is not None:
                dot.edge(parent_id, node_id, label=edge_label)
            return node_id

        def traverse(node, parent_id=None, edge_label=""):
            if node is None:
                return
            current_id = add_node(node, parent_id, edge_label)
            if node.value is None:  # 内部节点继续递归
                traverse(node.left, current_id, "是")
                traverse(node.right, current_id, "否")

        traverse(self.root)
        return dot

    def plot_tree_in_memory(self, figsize=(8, 6)):
        """
        在内存中渲染决策树并用 matplotlib 显示 (不产生中间文件)

        Parameters
        ----------
        figsize : tuple, default=(8, 6)
            图片显示尺寸
        """
        dot = self.export_graphviz()
        png_bytes = dot.pipe(format="png")  # 内存中生成PNG
        img = mpimg.imread(io.BytesIO(png_bytes))
        plt.figure(figsize=figsize)
        plt.imshow(img)
        plt.axis("off")
        plt.tight_layout()
        plt.show()


def plot_sklearn_tree_in_memory(
    sklearn_tree, feature_names, class_names=None, figsize=(8, 6)
):
    """用 graphviz 在内存中渲染 sklearn 决策树，与自研树风格一致"""
    if class_names is None:
        class_names = ["否", "是"]
    dot_data = sk_export_graphviz(
        sklearn_tree,
        out_file=None,
        feature_names=feature_names,
        class_names=class_names,
        filled=True,
        rounded=True,
        special_characters=True,
        fontname="SimHei",
    )
    dot = graphviz.Source(dot_data)
    png_bytes = dot.pipe(format="png")
    img = mpimg.imread(io.BytesIO(png_bytes))
    plt.figure(figsize=figsize)
    plt.imshow(img)
    plt.axis("off")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # 加载数据并预处理
    if input("Input not null: watermelon2.csv, if null: watermelon3.csv\n"):
        tree = DecisionTree("data/watermelon2.csv")
    else:
        tree = DecisionTree("data/watermelon3.csv")
    tree.data_processing()

    # 划分训练集和测试集 (12 : 5), 使用分层抽样
    X_train, X_test, y_train, y_test = train_test_split(
        tree.X, tree.y, test_size=5, random_state=0, stratify=tree.y
    )

    # ==================== 自研 CART 无限制深度 ====================
    tree.fit(X_train, y_train, max_depth=None, min_samples_split=2)
    print("\n" + "=" * 50)
    print("自研 CART 无限制深度")
    print(
        f"深度: {tree.get_depth()}, 叶节点: {tree.get_n_leaves()}, 总节点: {tree.get_n_nodes()}"
    )
    print(f"训练集准确率: {tree.score(X_train, y_train):.4f}")
    print(f"测试集准确率: {tree.score(X_test, y_test):.4f}")
    tree.plot_tree_in_memory(figsize=(8, 6))  # 可视化无限制树

    # ==================== 自研 CART 预剪枝 ====================
    tree.fit(X_train, y_train, max_depth=2, min_samples_split=2)
    print("\n" + "=" * 50)
    print("自研 CART 预剪枝")
    print(
        f"深度: {tree.get_depth()}, 叶节点: {tree.get_n_leaves()}, 总节点: {tree.get_n_nodes()}"
    )
    print(f"训练集准确率: {tree.score(X_train, y_train):.4f}")
    print(f"测试集准确率: {tree.score(X_test, y_test):.4f}")
    tree.plot_tree_in_memory(figsize=(8, 6))  # 可视化预剪枝树

    # ==================== sklearn 对比 ====================
    # sklearn 无限制深度
    clf = DecisionTreeClassifier(
        criterion="gini", max_depth=None, min_samples_split=2, random_state=0
    )
    clf.fit(X_train, y_train)
    print("\n" + "=" * 50)
    print("sklearn CART 无限制深度")
    print(
        f"深度: {clf.get_depth()}, 叶节点: {clf.get_n_leaves()}, 总节点: {clf.tree_.node_count}"
    )
    print(f"训练集准确率: {clf.score(X_train, y_train):.4f}")
    print(f"测试集准确率: {clf.score(X_test, y_test):.4f}")
    plot_sklearn_tree_in_memory(
        clf, tree.feature_names, figsize=(8, 6)
    )  # 可视化无限制树

    # sklearn 预剪枝
    clf = DecisionTreeClassifier(
        criterion="gini", max_depth=2, min_samples_split=2, random_state=0
    )
    clf.fit(X_train, y_train)
    print("\n" + "=" * 50)
    print("sklearn CART 预剪枝")
    print(
        f"深度: {clf.get_depth()}, 叶节点: {clf.get_n_leaves()}, 总节点: {clf.tree_.node_count}"
    )
    print(f"训练集准确率: {clf.score(X_train, y_train):.4f}")
    print(f"测试集准确率: {clf.score(X_test, y_test):.4f}")
    plot_sklearn_tree_in_memory(
        clf, tree.feature_names, figsize=(8, 6)
    )  # 可视化预剪枝树
