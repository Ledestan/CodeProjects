"""
项目名称: 人脸识别分析
创建日期: 2026-05-14
需求文件: data

依赖库:
opencv-python>=4.12.0.88
matplotlib>=3.10.8
numpy>=2.2.6
scikit-learn>=1.8.0
"""

import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler

np.random.seed(0)
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False


class FaceRecAnalysis:
    def __init__(self, path, img_size):
        self.path = path
        self.img_size = img_size
        self.n_show, self.rows, self.cols = 15, 3, 5

        # 数据加载与预处理
        self.img_shape = (self.img_size[1], self.img_size[0])  # 数据形状
        self.X = None  # 原始图像数据矩阵
        self.y = None  # 标签向量
        self.scaler = None  # 标准化器
        self.pca = None  # PCA 模型
        self.X_scaler = None  # 标准化后的数据

        # PCA 分析
        self.explained_var = None  # 主成分方差比
        self.cumulative_var = None  # 累计方差比
        self.X_pca = None  # 降维坐标
        self.eigenfaces = None  # 特征脸
        self.k = None  # 主成分

        # LDA 分析
        self.fisher_faces = None  # Fisher 脸矩阵

        # PCA + LDA 级联
        self.pca_for_lda = None  # PCA 模型
        self.lda = None  # LDA 模型

    def load_data(self):
        """加载 ORL 人脸数据集并预处理"""
        X, y = [], []

        if not os.path.exists(self.path):
            raise FileNotFoundError(f"数据路径 {self.path} 不存在")

        # 获取所有子文件夹并排序, 确保ID顺序一致
        person_dirs = [
            d
            for d in os.listdir(self.path)
            if os.path.isdir(os.path.join(self.path, d))
        ]
        person_dirs.sort()

        # 遍历数据集文件夹
        for label, person_dir_name in enumerate(person_dirs, start=1):
            person_dir = os.path.join(self.path, person_dir_name)

            # 读取该文件夹下所有图片
            img_files = [
                f
                for f in os.listdir(person_dir)
                if f.lower().endswith((".jpg", ".jpeg", ".png", ".pgm"))
            ]
            img_files.sort()  # 排序

            for img_name in img_files:
                img_path = os.path.join(person_dir, img_name)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

                if img is None:
                    print(f"警告: 无法读取图像 {img_path}")
                    continue

                img = cv2.resize(img, self.img_size)  # 统一尺寸
                X.append(img.flatten())  # 展平
                y.append(label)  # 使用顺序标签, 避免文件夹名解析错误

        if len(X) == 0:
            raise ValueError("未加载到任何图像数据, 请检查数据集路径")

        self.X = np.array(X, dtype=np.float32)
        self.y = np.array(y)

        # 标准化处理
        self.scaler = StandardScaler()
        self.X_scaler = self.scaler.fit_transform(self.X)

    def data_preview(self):
        """数据预览"""
        # 打印基础信息
        n_samples, n_features = self.X.shape

        # 将标签转换为集合并排序, 获取所有不重复的类别ID
        unique_labels = sorted(set(self.y))
        n_classes = len(unique_labels)

        print("\n" + "=" * 50)
        print(f"样本总数: {n_samples}")
        print(f"特征维度: {n_features}")
        print(f"标签种类数: {n_classes}")
        print(f"标签范围: {unique_labels[0]} - {unique_labels[-1]}")

        # 随机取样
        n_samples = self.X.shape[0]
        random_indices = np.random.choice(n_samples, 16, replace=False)

        plt.figure(figsize=(6, 6))

        for i, idx in enumerate(random_indices):
            plt.subplot(4, 4, i + 1)

            img = self.X[idx].reshape(self.img_shape)  # 维度还原

            plt.imshow(img, cmap="gray")
            plt.title(f"ID: {self.y[idx]}")
            plt.axis("off")  # 隐藏坐标轴

        plt.suptitle("随机抽样 16 张原始人脸预览", fontsize=16)
        plt.tight_layout()
        plt.show()

    def pca_analyze(self):
        """PCA 分析"""
        # 执行 PCA
        n_samples, n_features = self.X_scaler.shape
        self.pca = PCA(n_components=min(n_samples - 1, n_features))
        self.pca.fit(self.X_scaler)

        self.explained_var = self.pca.explained_variance_ratio_
        self.cumulative_var = np.cumsum(self.explained_var)
        self.X_pca = self.pca.transform(self.X_scaler)

        # 提取特征脸
        self.eigenfaces = [
            comp.reshape(self.img_shape) for comp in self.pca.components_
        ]

        # 可视化部分
        self._plot_eigenfaces()
        self.k = self._plot_cumulative_variance()
        self._plot_reconstruction(n_samples=5)

    def _plot_eigenfaces(self):
        """展示前 n_show 张特征脸"""
        fig, axes = plt.subplots(
            self.rows, self.cols, figsize=(2 * self.cols, 2 * self.rows)
        )
        axes = axes.flatten()

        for i in range(self.n_show):
            axes[i].imshow(self.eigenfaces[i], cmap="gray")
            axes[i].set_title(f"PCA {i+1}")
            axes[i].axis("off")

        for j in range(self.n_show, len(axes)):
            axes[j].axis("off")

        fig.suptitle(f"前 {self.n_show} 张特征脸 (Eigenfaces)", fontsize=12)
        plt.tight_layout()
        plt.show()

    def _plot_cumulative_variance(self):
        """绘制累计方差曲线并返回 k_95"""
        x = np.arange(1, len(self.cumulative_var) + 1)
        k_90 = np.argmax(self.cumulative_var >= 0.9) + 1
        k_95 = np.argmax(self.cumulative_var >= 0.95) + 1

        plt.figure(figsize=(8, 6))
        plt.plot(x, self.cumulative_var, "b-", linewidth=2, label="累计解释方差")
        plt.axhline(y=0.9, color="r", linestyle="--", label="90%")
        plt.axhline(y=0.95, color="g", linestyle="--", label="95%")
        plt.axvline(x=k_90, color="r", linestyle=":", alpha=0.7)
        plt.axvline(x=k_95, color="g", linestyle=":", alpha=0.7)
        plt.scatter([k_90, k_95], [0.9, 0.95], color="black", zorder=5)
        plt.text(k_90, 0.91, f"k={k_90}", ha="center", va="bottom")
        plt.text(k_95, 0.96, f"k={k_95}", ha="center", va="bottom")
        plt.xlabel("主成分数量")
        plt.ylabel("累计解释方差比例")
        plt.title("PCA 累计方差曲线")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.show()

        print("\n" + "=" * 50)
        print(f"达到 90% 所需主成分数: {k_90}")
        print(f"达到 95% 所需主成分数: {k_95}")
        return k_95

    def _plot_reconstruction(self, n_samples=5):
        """人脸重构对比: 随机 n_samples 张原图与 k 个主成分重构"""
        indices = np.random.choice(self.X.shape[0], n_samples, replace=False)

        fig, axes = plt.subplots(2, n_samples, figsize=(10, 4))

        # 使用已经训练好的模型取前 k 个成分进行降维和重构, 直接切片 components_ 和 mean_ 即可
        components_k = self.pca.components_[: self.k]
        mean_k = self.pca.mean_

        # 提取对应随机索引的标准化数据
        x_selected = self.X_scaler[indices]

        # 中心化后投影到前 k 个主成分, 再逆变换回来
        x_centered = x_selected - self.pca.mean_
        x_pca_k = np.dot(x_centered, components_k.T)
        x_recon_centered = np.dot(x_pca_k, components_k) + mean_k

        # 逆标准化还原到原始像素范围
        x_recon = self.scaler.inverse_transform(x_recon_centered)

        for i, idx in enumerate(indices):
            # 原图
            original = self.X[idx].reshape(self.img_shape)
            axes[0, i].imshow(original, cmap="gray")
            axes[0, i].set_title(f"原图 (ID:{self.y[idx]})")
            axes[0, i].axis("off")

            # 重构图
            recon_img = x_recon[i].reshape(self.img_shape)
            axes[1, i].imshow(recon_img, cmap="gray")
            axes[1, i].set_title(f"重构 (k={self.k})")
            axes[1, i].axis("off")

        fig.suptitle(f"PCA 人脸重构对比 (k={self.k})", fontsize=14)
        plt.tight_layout()
        plt.show()

    def analyze_reconstruction_error(self):
        """
        重构误差分析

        降维再还原: 对每个 k (10, 30, 50, 100), 用 PCA 把图像压缩到 k 维, 再解压回原尺寸.

        反标准化: 用之前保存的 scaler, 把解压后的数据变回正常像素值 (0~255), 不然算出的误差没意义.

        算误差并画图: 计算所有像素的平均误差 (MSE), 画一条 k-MSE 曲线, k 越大误差越小.
        """
        k_values = [10, 30, 50, 100]
        mse_list = []

        print("\n" + "=" * 50)
        for k in k_values:
            # 用指定主成分数重新拟合 PCA
            pca_k = PCA(n_components=k)
            X_scaled_k = pca_k.fit_transform(self.X_scaler)
            # 逆变换回到标准化空间
            X_scaled_recon = pca_k.inverse_transform(X_scaled_k)
            # 反标准化回原始像素值范围
            X_recon = self.scaler.inverse_transform(X_scaled_recon)
            # 计算与原始图像的均方误差
            mse = np.mean((self.X - X_recon) ** 2)
            mse_list.append(mse)
            print(f"k = {k:3d}, 平均 MSE = {mse:.4f}")

        # 绘制 k 值与 MSE 关系曲线
        plt.figure(figsize=(8, 5))
        plt.plot(k_values, mse_list, "bo-", linewidth=2, markersize=8)
        plt.xlabel("主成分数量 k")
        plt.ylabel("均方误差 (MSE)")
        plt.title("PCA 重构误差随 k 值的变化")
        plt.grid(alpha=0.3)
        # 在每个点上标注 MSE 数值
        for k, mse in zip(k_values, mse_list):
            plt.text(k, mse, f"{mse:.2f}", ha="center", va="bottom")
        plt.show()

    def lda_analyze(self):
        """LDA 分析: LDA 可视化 (降维散点图与 Fisher 脸)"""
        # 训练全量数据上的 LDA
        n_classes = len(np.unique(self.y))
        lda = LinearDiscriminantAnalysis(n_components=n_classes - 1)
        lda.fit(self.X_scaler, self.y)

        # 交叉验证准确率
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(
            lda, self.X_scaler, self.y, cv=cv, scoring="accuracy"
        )
        print("\n" + "=" * 50)
        print("LDA 分析可视化")
        print(
            f"五折交叉验证准确率: {cv_scores.mean() * 100:.2f}% (+/- {cv_scores.std() * 100:.2f}%)"
        )

        # 计算训练集准确率
        y_pred_train = lda.predict(self.X_scaler)
        train_acc = accuracy_score(self.y, y_pred_train)
        print(f"训练集准确率: {train_acc * 100:.2f}%")

        # LDA 降维散点图 (前两个分量)
        X_lda = lda.transform(self.X_scaler)
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(
            X_lda[:, 0], X_lda[:, 1], c=self.y, cmap="tab20", alpha=0.7, s=20
        )
        plt.colorbar(scatter, ticks=np.unique(self.y), label="Person ID")
        plt.xlabel("LDA 分量 1")
        plt.ylabel("LDA 分量 2")
        plt.title("LDA 投影 (前两个维度)")
        plt.grid(alpha=0.3)
        plt.show()

        # Fisher 脸 (直接取 scalings_ 的列)
        fig, axes = plt.subplots(
            self.rows, self.cols, figsize=(2 * self.cols, 2 * self.rows)
        )
        axes = axes.flatten()

        for i in range(self.n_show):
            # scalings_ 形状 (4096, 39), 每一列是一个 Fisher 脸向量
            fisher_face = lda.scalings_[:, i].reshape(self.img_shape)
            axes[i].imshow(fisher_face, cmap="gray")
            axes[i].set_title(f"Fisher {i+1}")
            axes[i].axis("off")

        for j in range(self.n_show, len(axes)):
            axes[j].axis("off")

        fig.suptitle("LDA 提取的 Fisher 脸", fontsize=12)
        plt.tight_layout()
        plt.show()

    def mixed_analyze(self):
        """PCA + LDA 级联分析"""
        # 先用 PCA 降维 (样本数 - 类别数)
        n_pca = self.X.shape[0] - len(np.unique(self.y))
        self.pca_for_lda = PCA(n_components=n_pca)
        X_pca = self.pca_for_lda.fit_transform(self.X_scaler)

        # 再用 LDA 降维 (类别数 - 1)
        n_lda = len(np.unique(self.y)) - 1
        self.lda = LinearDiscriminantAnalysis(n_components=n_lda)
        X_lda = self.lda.fit_transform(X_pca, self.y)

        # 提取 Fisher 脸: 将 LDA 方向映射回原始像素空间
        pca_components = (
            self.pca_for_lda.components_
        )  # pca_components 形状: (n_pca, 4096)
        lda_weights = self.lda.scalings_  # lda_scalings 形状: (n_pca, n_lda)

        # 矩阵乘法: (4096, n_pca) x (n_pca, n_lda) -> (4096, n_lda)
        fisher_faces_matrix = pca_components.T @ lda_weights
        # 每一列是一个 Fisher 脸, reshape 成图像尺寸
        fisher_faces = [
            fisher_faces_matrix[:, i].reshape(self.img_shape) for i in range(n_lda)
        ]
        self.fisher_faces = fisher_faces  # Fisher 脸矩阵

        # 可视化前 Fisher 脸
        fig, axes = plt.subplots(
            self.rows, self.cols, figsize=(2 * self.cols, 2 * self.rows)
        )
        axes = axes.flatten()

        for i in range(self.n_show):
            axes[i].imshow(fisher_faces[i], cmap="gray")
            axes[i].set_title(f"Fisher {i+1}")
            axes[i].axis("off")

        for j in range(self.n_show, len(axes)):
            axes[j].axis("off")

        fig.suptitle(f"前 {self.n_show} 张 Fisher 脸", fontsize=12)
        plt.tight_layout()
        plt.show()

        # 绘制前两个 LDA 分量的散点图
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(
            X_lda[:, 0], X_lda[:, 1], c=self.y, cmap="tab20", alpha=0.7, edgecolors="k"
        )
        plt.colorbar(scatter, ticks=range(0, 40), label="人员 ID")
        plt.xlabel("LDA 分量 1")
        plt.ylabel("LDA 分量 2")
        plt.title("PCA + LDA 投影前两个分量")
        plt.grid(alpha=0.3)
        plt.show()

    def comparative_analyze(self):
        """对比分析"""
        # 原始人脸与 PCA 重构人脸对比
        pca_recon = PCA(n_components=self.k)
        pca_recon.fit(self.X_scaler)

        n_show = 5
        indices = np.random.choice(self.X.shape[0], n_show, replace=False)
        fig, axes = plt.subplots(2, n_show, figsize=(10, 5))
        for i, idx in enumerate(indices):
            # 原始图像
            original = self.X[idx].reshape(self.img_shape)
            axes[0, i].imshow(original, cmap="gray")
            axes[0, i].set_title(f"原始 (ID:{self.y[idx]})")
            axes[0, i].axis("off")

            # PCA 重构
            x_scaled = self.X_scaler[idx].reshape(1, -1)
            x_recon = self.scaler.inverse_transform(
                pca_recon.inverse_transform(pca_recon.transform(x_scaled))
            ).reshape(self.img_shape)
            axes[1, i].imshow(x_recon, cmap="gray")
            axes[1, i].set_title(f"PCA重构 k={self.k}")
            axes[1, i].axis("off")
        fig.suptitle("原始人脸 vs PCA 重构人脸", fontsize=14)
        plt.tight_layout()
        plt.show()

        # 特征脸 vs Fisher 脸对比
        n_show = 10
        fig = plt.figure(figsize=(8, 6))

        # 上半部分: 特征脸
        for i in range(n_show):
            ax = fig.add_subplot(4, self.cols, i + 1)
            ax.imshow(self.eigenfaces[i], cmap="gray")
            ax.set_title(f"特征脸 {i+1}", fontsize=10)
            ax.axis("off")

        # 下半部分: Fisher 脸
        for i in range(n_show):
            ax = fig.add_subplot(4, self.cols, n_show + i + 1)
            if i < len(self.fisher_faces):
                ax.imshow(self.fisher_faces[i], cmap="gray")
                ax.set_title(f"Fisher脸 {i+1}", fontsize=10)
            ax.axis("off")

        fig.suptitle("PCA 特征脸 vs LDA Fisher 脸", fontsize=12)
        plt.tight_layout()
        plt.show()


class FaceVerifier:
    """基于 LDA 投影和余弦相似度的人脸验证器"""

    def __init__(self, scaler, pca, lda, img_size=(64, 64)):
        """
        scaler: 训练好的 StandardScaler 对象
        pca:    训练好的 PCA 对象 (用于 PCA + LDA 级联)
        lda:    训练好的 LinearDiscriminantAnalysis 对象
        img_size: 图像缩放尺寸 (宽, 高), 默认 (64,64)
        """
        self.scaler = scaler
        self.pca = pca
        self.lda = lda
        self.img_size = img_size
        self.threshold = None  # 余弦相似度阈值

    def _preprocess_and_project(self, img_path):
        """读取单张人脸图片, 返回其在 LDA 空间中的特征向量"""
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"无法读取图像: {img_path}")
        img = cv2.resize(img, self.img_size)
        x = img.flatten().astype(np.float32).reshape(1, -1)
        x_std = self.scaler.transform(x)
        if self.pca is not None:
            x_std = self.pca.transform(x_std)
        x_lda = self.lda.transform(x_std)
        return x_lda.flatten()

    @staticmethod
    def cosine_similarity(v1, v2):
        """计算两个向量的余弦相似度"""
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return np.dot(v1, v2) / (norm1 * norm2)

    def set_threshold(self, threshold):
        """手动设置相似度阈值 (0~1), 大于等于该阈值判定为同一人"""
        self.threshold = threshold

    def estimate_threshold(self, X_lda, labels, far_tolerance=0.01):
        """
        根据目标 FAR (False Accept Rate) 自动估计阈值。
        far_tolerance: 允许的最大错误接受率 (0~1), 默认 0.01 表示最多 1% 的异类被判为同一人。
        阈值设定方法: 找出一个值, 使得训练集中异类配对余弦相似度高于该阈值的比例不超过 far_tolerance。
        """
        # 收集所有异类配对的余弦相似度
        inter_sim = []
        unique_labels = np.unique(labels)
        n_labels = len(unique_labels)
        for i in range(n_labels):
            idx_a = np.where(labels == unique_labels[i])[0]
            for j in range(i + 1, n_labels):
                idx_b = np.where(labels == unique_labels[j])[0]
                for a in idx_a:
                    for b in idx_b:
                        sim = self.cosine_similarity(X_lda[a], X_lda[b])
                        inter_sim.append(sim)
        if len(inter_sim) == 0:
            raise ValueError("训练集中没有异类样本可用于估计阈值")
        inter_sim = np.array(inter_sim)
        # 按升序排列
        inter_sim_sorted = np.sort(inter_sim)
        # 找到第 (1 - far_tolerance) 百分位数作为阈值, 使得高于它的比例 <= far_tolerance
        # 即 threshold = 第 100*(1 - far_tolerance) 百分位数
        percentile = 100 * (1 - far_tolerance)
        self.threshold = np.percentile(inter_sim_sorted, percentile)
        # 打印信息
        false_accepts = np.sum(inter_sim_sorted >= self.threshold)
        actual_far = false_accepts / len(inter_sim_sorted)
        print("\n" + "=" * 50)
        print(f"为满足 FAR <= {far_tolerance:.4f}, 设定阈值 = {self.threshold:.4f}")
        print(f"训练集上的实际 FAR: {actual_far:.4f}")

    def verify(self, img_path1, img_path2, threshold=None):
        """
        判断两张人脸图片是否属于同一个人
        返回: (is_same: bool, similarity: float)
        """
        vec1 = self._preprocess_and_project(img_path1)
        vec2 = self._preprocess_and_project(img_path2)
        sim = self.cosine_similarity(vec1, vec2)

        thr = threshold if threshold is not None else self.threshold
        if thr is None:
            raise ValueError("阈值未设置, 请先调用 set_threshold 或 estimate_threshold")
        is_same = sim >= thr
        return is_same, sim


if __name__ == "__main__":
    # 初始化并加载数据
    FaceRec = FaceRecAnalysis("data", (64, 64))
    FaceRec.load_data()

    # 执行分析流程
    FaceRec.data_preview()
    FaceRec.pca_analyze()
    FaceRec.lda_analyze()
    FaceRec.mixed_analyze()

    # 获取训练数据的 LDA 投影 (用于阈值估计)
    X_pca_train = FaceRec.pca_for_lda.transform(FaceRec.X_scaler)
    X_lda_train = FaceRec.lda.transform(X_pca_train)

    # 创建验证器
    verifier = FaceVerifier(
        scaler=FaceRec.scaler,
        pca=FaceRec.pca_for_lda,
        lda=FaceRec.lda,
        img_size=(64, 64),
    )

    # 设定阈值, 目标 FAR=0.01
    verifier.estimate_threshold(X_lda_train, FaceRec.y, far_tolerance=0.01)

    # 进行人脸验证测试
    # 请根据你实际的文件名修改路径
    try:
        # 测试 1: 同一个人
        is_same_A, sim_A = verifier.verify("data/s1/1.pgm", "data/s1/2.pgm")
        print(f"测试A - 同一个人: {'是' if is_same_A else '否'} (相似度: {sim_A:.4f})")

        # 测试 2: 不同的人
        is_same_B, sim_B = verifier.verify("data/s1/1.pgm", "data/s2/1.pgm")
        print(f"测试B - 不同的人: {'是' if is_same_B else '否'} (相似度: {sim_B:.4f})")

    except Exception as e:
        print(f"测试出错: {e}")
        print("请检查图片路径是否正确, 或者 data 文件夹下是否有图片.")
