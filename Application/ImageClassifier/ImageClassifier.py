"""
项目名称: 图像识别分类器
创建日期: 2026-04-22

包含:
- ImageClassifier: LBP 分块直方图 + Sigmoid 非线性分类器
- Window: 图形界面应用程序 (GUI)

需求文件: data/train, data/test

依赖库:
numpy>=2.2.6
Pillow>=11.0.0
"""

import os
import sys
import tkinter as tk
from tkinter import filedialog, messagebox

import numpy as np
from PIL import Image

sys.dont_write_bytecode = True


class ImageClassifier:
    """
    核心步骤:
        Step1: 使用分块 LBP 直方图将 224*224 灰度图转化为纹理特征向量
        Step2: 构建 Sigmoid 非线性模型: y = 1/(1+e^{-(w·x+b)})
        Step3: 定义交叉熵损失函数
        Step4: 计算损失对参数的梯度
        Step5: 梯度下降迭代训练
        Step6: 模型测试与单张预测
    """

    def __init__(self, grid_rows=4, grid_cols=4, lbp_bins=256):
        """
        初始化分类器。
        参数:
            grid_rows: 垂直方向分块数
            grid_cols: 水平方向分块数
            lbp_bins: LBP 直方图的 bin 数（原始 LBP 为 256）
        """
        self.img_size = (224, 224)
        self.H, self.W = 224, 224
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.lbp_bins = lbp_bins

        # 特征维度
        self.feature_dim = grid_rows * grid_cols * lbp_bins

        # 模型参数 (待训练)
        self.w = None  # shape (feature_dim,)
        self.b = None  # 标量

        # 训练历史记录
        self.loss_history = []

        # 特征标准化参数（训练时计算，预测时使用）
        self.feature_mean = None
        self.feature_std = None

    # ==================== 数据加载与预处理 ====================
    def load_images(self, data_dir: str):
        """
        从指定目录加载猫狗图片

        参数:
            data_dir: str, 数据根目录

        返回:
            images: list of np.ndarray, 每张图片的灰度矩阵 (H, W), dtype=float64
            labels: list of int, 对应标签 (1=猫, 0=狗)
        """
        images = []
        labels = []
        # cat 标签 1, dog 标签 0
        for class_name, label in [("cat", 1), ("dog", 0)]:
            class_dir = os.path.join(data_dir, class_name)
            if not os.path.isdir(class_dir):
                print(f"警告: 目录 {class_dir} 不存在，跳过")
                continue
            for fname in sorted(os.listdir(class_dir)):
                if fname.lower().endswith((".jpg", ".png", ".jpeg")):
                    img_path = os.path.join(class_dir, fname)
                    img = self.load_single_image(img_path)
                    if img is not None:
                        images.append(img)
                        labels.append(label)
        print(
            f"从 '{data_dir}' 加载了 {len(images)} 张图片 (猫: {labels.count(1)}, 狗: {labels.count(0)})"
        )
        return images, labels

    def load_single_image(self, path: str):
        """加载单张图片，转为灰度，缩放至指定尺寸"""
        try:
            img = Image.open(path).convert("L")  # 灰度
            img = img.resize(self.img_size, Image.Resampling.LANCZOS)
            return np.array(img, dtype=np.float64)
        except Exception as e:
            print(f"无法加载图片 '{path}': {e}")
            return None

    # ==================== Step1: 分块 LBP 直方图特征提取 ====================
    def extract_features(self, img):
        """
        对单张灰度图片计算分块 LBP 直方图特征。
        返回形状为 (self.feature_dim,) 的归一化直方图向量。
        """
        # 确保图像是 uint8 类型
        if img.dtype != np.uint8:
            img = img.astype(np.uint8)

        H, W = img.shape
        block_h = H // self.grid_rows
        block_w = W // self.grid_cols

        # 计算整张图片的原始 LBP (8 邻域)
        lbp_img = np.zeros_like(img, dtype=np.uint8)
        center = img[1:-1, 1:-1]

        # 8 个邻域，按顺时针从左上开始
        neighbors = [
            img[0:-2, 0:-2],  # 左上
            img[0:-2, 1:-1],  # 上
            img[0:-2, 2:],  # 右上
            img[1:-1, 2:],  # 右
            img[2:, 2:],  # 右下
            img[2:, 1:-1],  # 下
            img[2:, 0:-2],  # 左下
            img[1:-1, 0:-2],  # 左
        ]

        for p, neigh in enumerate(neighbors):
            lbp_img[1:-1, 1:-1] += ((neigh >= center) * (1 << p)).astype(np.uint8)

        # 按块统计直方图，并归一化
        features = []
        for i in range(self.grid_rows):
            for j in range(self.grid_cols):
                block = lbp_img[
                    i * block_h : (i + 1) * block_h, j * block_w : (j + 1) * block_w
                ].ravel()
                hist, _ = np.histogram(block, bins=self.lbp_bins, range=(0, 255))
                # 归一化为频率，避免块大小不一致影响
                features.extend(hist.astype(np.float64) / block.size)

        return np.array(features)

    def prepare_dataset(self, images):
        """
        将图片列表转换为特征矩阵

        参数:
            images: list of np.ndarray

        返回:
            features: np.ndarray, shape (N, feature_dim)
        """
        features = [self.extract_features(img) for img in images]
        return np.array(features)

    # ==================== 特征标准化 (Z-score) ====================
    def fit_scaler(self, features):
        """根据训练特征计算均值和标准差（训练时调用）"""
        self.feature_mean = np.mean(features, axis=0)
        self.feature_std = np.std(features, axis=0) + 1e-8  # 防止除零

    def transform(self, features):
        """对特征进行标准化（均值0，标准差1）"""
        if self.feature_mean is None or self.feature_std is None:
            raise ValueError("请先调用 fit_scaler 计算标准化参数")
        return (features - self.feature_mean) / self.feature_std

    # ==================== Step2: 非线性模型 (Sigmoid) ====================
    @staticmethod
    def sigmoid(t):
        """
        Sigmoid 激活函数: y = 1 / (1 + e^{-t})
        使用数值裁剪防止溢出
        """
        t = np.clip(t, -500, 500)  # exp(500) 接近浮点数上限
        return 1.0 / (1.0 + np.exp(-t))

    def forward(self, features):
        """
        前向传播: 计算预测概率

        公式: t = w·x + b,  y = sigmoid(t)

        参数:
            features: np.ndarray, shape (N, D) 或 (D,)

        返回:
            probs: np.ndarray, 预测概率 (0~1)
        """
        if self.w is None or self.b is None:
            raise ValueError("模型尚未训练或加载参数，请先训练或调用 load_params")
        t = np.dot(features, self.w) + self.b
        return self.sigmoid(t)

    # ==================== Step3: 损失函数 (交叉熵) ====================
    @staticmethod
    def cross_entropy_loss(y_true, y_pred):
        """
        二分类交叉熵损失
        L = -[z*ln(y) + (1-z)*ln(1-y)]

        参数:
            y_true: np.ndarray, 真实标签 (0/1)
            y_pred: np.ndarray, 预测概率

        返回:
            loss: float, 平均损失
        """
        eps = 1e-12
        y_pred = np.clip(y_pred, eps, 1 - eps)
        losses = -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return np.mean(losses)

    # ==================== Step4: 梯度计算 ====================
    def compute_gradients(self, features, y_true):
        """
        计算损失对 w, b 的梯度

        对交叉熵 + Sigmoid 组合，简化后为:
            dL/dw = (1/N) * (y_pred - y_true) * x
            dL/db = (1/N) * (y_pred - y_true)

        参数:
            features: np.ndarray, shape (N, D)
            y_true: np.ndarray, shape (N,)

        返回:
            dw: np.ndarray, shape (D,)
            db: float
        """
        y_pred = self.forward(features)
        error = y_pred - y_true  # (N,)

        dw = np.mean(error[:, np.newaxis] * features, axis=0)  # (D,)
        db = np.mean(error)
        return dw, db

    # ==================== Step5: 梯度下降训练 ====================
    def train(
        self, features, labels, learning_rate, tolerance, max_iters, verbose=True
    ):
        """
        使用梯度下降训练模型参数

        参数:
            features: np.ndarray, shape (N, D)
            labels: np.ndarray, shape (N,) 真实标签 0/1
            learning_rate: 学习率
            tolerance: 收敛容差 (相邻损失差值小于此值时停止)
            max_iters: 最大迭代次数
            verbose: 是否打印详细日志

        返回:
            w_opt: np.ndarray, 最优权重
            b_opt: float, 最优偏置
            final_loss: float, 最终损失值
        """
        # ---------- 特征标准化 ----------
        self.fit_scaler(features)
        features = self.transform(features)

        # 参数初始化 (小随机数)
        np.random.seed(0)
        self.w = np.random.randn(self.feature_dim) * 0.01
        self.b = 0.0

        self.loss_history = []
        prev_loss = float("inf")
        final_loss = prev_loss

        for i in range(max_iters):
            # 前向传播计算损失
            y_pred = self.forward(features)
            loss = self.cross_entropy_loss(labels, y_pred)
            self.loss_history.append(loss)

            # 收敛检查
            if abs(prev_loss - loss) < tolerance:
                final_loss = loss
                if verbose:
                    print(f"收敛于第 {i} 次迭代，损失变化 < {tolerance}")
                break

            # 梯度计算与参数更新
            dw, db = self.compute_gradients(features, labels)
            self.w -= learning_rate * dw
            self.b -= learning_rate * db

            prev_loss = loss
            final_loss = loss

            # 定期打印
            if verbose and (i % 100 == 0 or i < 10):
                print(f"Iter {i:4d} | Loss: {loss:.6f} | b: {self.b:.6f}")

        if verbose:
            print(f"\n训练完成！")
            print(f"最优参数: w 的前 5 维: {self.w[:5]}, b = {self.b:.6f}")
            print(f"最终损失: {final_loss:.6f}")

        return self.w.copy(), self.b, final_loss

    # ==================== Step6: 测试与评估 ====================
    def evaluate(self, test_dir):
        """
        在测试集上评估模型性能

        参数:
            test_dir: str, 测试集根目录 (结构同训练集: test_dir/cat, test_dir/dog)

        返回:
            accuracy: float, 准确率
            results: list of dict, 每张图片的详细预测结果
        """
        if self.w is None or self.b is None:
            raise ValueError("模型尚未训练，无法评估")
        if self.feature_mean is None or self.feature_std is None:
            raise ValueError("标准化参数未设置，请先训练或加载标准化参数")

        test_imgs, test_labels = self.load_images(test_dir)
        if not test_imgs:
            return 0.0, []

        test_features = self.prepare_dataset(test_imgs)
        test_features = self.transform(test_features)  # 标准化
        probs = self.forward(test_features)  # 预测概率
        preds = (probs >= 0.5).astype(int)  # 阈值 0.5 判定类别

        correct = np.sum(preds == test_labels)
        accuracy = correct / len(test_labels)

        results = []
        for i, (prob, pred, true_label) in enumerate(zip(probs, preds, test_labels)):
            results.append(
                {
                    "index": i,
                    "true_label": true_label,
                    "true_name": "猫" if true_label == 1 else "狗",
                    "pred_prob": prob,
                    "pred_label": pred,
                    "pred_name": "猫" if pred == 1 else "狗",
                    "correct": (pred == true_label),
                }
            )

        print(f"\n测试集评估结果:")
        print(f"总样本数: {len(test_labels)}")
        print(f"正确数: {correct}")
        print(f"准确率: {accuracy:.2%}")

        # 打印错误详情
        errors = [r for r in results if not r["correct"]]
        if errors:
            print(f"错误预测的样本 ({len(errors)} 个):")
            for err in errors:
                print(
                    f"图片 {err['index']}: 真实={err['true_name']}, 预测概率={err['pred_prob']:.4f} → {err['pred_name']}"
                )

        return accuracy, results

    def predict_single(self, img_path):
        """
        对单张图片进行预测（供 GUI 调用）

        参数:
            img_path: str, 图片文件路径

        返回:
            dict: {
                'probability': float,  预测为猫的概率
                'class_name': str,      '猫' 或 '狗'
                'class_label': int,     1 或 0
                'success': bool        是否成功预测
            }
        """
        if self.w is None or self.b is None:
            return {"success": False, "error": "模型未初始化"}
        if self.feature_mean is None or self.feature_std is None:
            return {"success": False, "error": "标准化参数未加载，请先加载训练好的模型"}

        img = self.load_single_image(img_path)
        if img is None:
            return {"success": False, "error": "无法加载图片"}

        features = self.extract_features(img).reshape(1, -1)
        features = self.transform(features)  # 标准化
        prob = self.forward(features)[0]
        pred_label = 1 if prob >= 0.5 else 0
        pred_name = "猫" if pred_label == 1 else "狗"

        return {
            "success": True,
            "probability": float(prob),
            "class_name": pred_name,
            "class_label": pred_label,
        }

    # ==================== 特征缓存功能 ====================
    def save_features(self, filepath, features, labels):
        """将提取好的特征和标签保存到 .npz 文件"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        np.savez(filepath, features=features, labels=labels)
        print(f"特征已缓存至: {filepath} | 形状: {features.shape}")

    def load_features(self, filepath):
        """
        从 .npz 文件加载特征和标签

        返回: (features, labels) 或 (None, None) 如果文件不存在
        """
        if os.path.exists(filepath):
            data = np.load(filepath)
            print(f"从缓存加载特征: {filepath} | 形状: {data['features'].shape}")
            return data["features"], data["labels"]
        else:
            print(f"未找到缓存文件: {filepath}")
            return None, None

    # ==================== 参数持久化 ====================
    def save_params(self, filepath):
        """保存模型参数及标准化参数到 .npz 文件"""
        if self.w is None or self.b is None:
            raise ValueError("没有可保存的参数，请先训练模型")
        if self.feature_mean is None or self.feature_std is None:
            raise ValueError("没有可保存的标准化参数，请先训练模型")
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        np.savez(
            filepath,
            w=self.w,
            b=self.b,
            feature_mean=self.feature_mean,
            feature_std=self.feature_std,
        )
        print(f"参数已保存至: {filepath}")

    def load_params(self, filepath):
        """从 .npz 文件加载模型参数及标准化参数"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"参数文件不存在: {filepath}")
        data = np.load(filepath)
        self.w = data["w"]
        self.b = data["b"].item()
        self.feature_mean = data["feature_mean"]
        self.feature_std = data["feature_std"]
        # 重新推断特征维度
        self.feature_dim = self.w.shape[0]
        print(f"参数已从 {filepath} 加载: w 维度 {self.w.shape}, b={self.b:.6f}")


class Window:
    """图形界面应用程序"""

    def __init__(self, root, classifier):
        self.root = root
        self.clf = classifier
        self.image_path = None

        root.title("图片分类器")
        root.geometry("500x300")
        root.resizable(False, False)

        # 标题
        title_label = tk.Label(root, text="图片分类器", font=("Arial", 16, "bold"))
        title_label.pack(pady=10)

        # 图片选择按钮
        self.btn_select = tk.Button(
            root, text="选择图片", command=self.select_image, width=15
        )
        self.btn_select.pack(pady=5)

        # 显示选中的文件名
        self.lbl_path = tk.Label(root, text="尚未选择图片", fg="gray", wraplength=400)
        self.lbl_path.pack(pady=5)

        # 预测按钮
        self.btn_predict = tk.Button(
            root, text="预测", command=self.predict, width=15, state=tk.DISABLED
        )
        self.btn_predict.pack(pady=5)

        # 结果显示区域
        self.lbl_result = tk.Label(root, text="", font=("Arial", 12))
        self.lbl_result.pack(pady=10)

        # 提示信息
        info_label = tk.Label(
            root, text="模型已加载，请选择一张图片进行预测。", fg="green"
        )
        info_label.pack(pady=10)

    def select_image(self):
        """打开文件对话框选择图片"""
        file_path = filedialog.askopenfilename(
            title="选择一张图片",
            filetypes=[("图片文件", "*.jpg *.jpeg *.png"), ("所有文件", "*.*")],
        )
        if file_path:
            self.image_path = file_path
            # 只显示文件名，避免路径过长
            self.lbl_path.config(text=os.path.basename(file_path), fg="black")
            self.btn_predict.config(state=tk.NORMAL)
            # 清空上次结果
            self.lbl_result.config(text="")

    def predict(self):
        """调用分类器预测选中图片"""
        if not self.image_path:
            return

        result = self.clf.predict_single(self.image_path)

        if not result["success"]:
            messagebox.showerror("错误", result.get("error", "预测失败"))
            return

        prob = result["probability"]
        class_name = result["class_name"]

        # 格式化显示结果
        self.lbl_result.config(
            text=f"预测结果: {class_name}\n概率 (猫=1, 狗=0): {prob:.4f}", fg="blue"
        )


def train_main():
    """分类器训练脚本"""
    # 初始化分类器
    clf = ImageClassifier()

    # 缓存文件路径
    cache_file = "models/train_features.npz"

    # 尝试从缓存加载特征
    X_train, y_train = clf.load_features(cache_file)

    if X_train is None:
        # 缓存不存在，需要加载图片并提取特征
        train_dir = "data/train"
        if not os.path.isdir(train_dir):
            print(f"错误：训练目录 '{train_dir}' 不存在。")
            return

        print("正在加载训练图片...")
        train_imgs, train_labels = clf.load_images(train_dir)
        if len(train_imgs) == 0:
            print("错误：训练集中未找到图片。")
            return

        print("正在提取 LBP 特征（首次较慢，后续将使用缓存）...")
        X_train = clf.prepare_dataset(train_imgs)
        y_train = np.array(train_labels)

        # 保存特征缓存
        clf.save_features(cache_file, X_train, y_train)
    else:
        print("已从缓存加载特征，跳过图片读取和特征提取。")

    # 训练模型
    print("开始训练...")
    w_opt, b_opt, final_loss = clf.train(
        X_train,
        y_train,
        learning_rate=0.1,
        tolerance=1e-5,
        max_iters=20000,
        verbose=True,
    )

    # 保存最优参数
    param_path = "models/params.npz"
    clf.save_params(param_path)

    # 输出总结
    print("\n" + "=" * 50)
    print(f"训练样本数: {len(y_train)}")
    print(f"最终损失: {final_loss:.6f}")
    print(f"参数已保存至: {param_path}")


def test_main():
    """分类器图形测试界面"""
    # 加载模型参数
    param_path = "models/params.npz"
    if not os.path.exists(param_path):
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror(
            "错误", f"未找到参数文件 '{param_path}'。\n请先运行 train.py 进行训练。"
        )
        return

    # 创建分类器实例并加载参数
    clf = ImageClassifier()
    clf.load_params(param_path)

    # 创建 GUI 窗口
    root = tk.Tk()
    app = Window(root, clf)
    root.mainloop()


if __name__ == "__main__":
    task_map = {
        "0": "Train",
        "1": "Test",
    }

    print("请选择要运行的任务：")
    print("  0 - Train (训练分类器)")
    print("  1 - Test  (图形界面测试)")

    choice = input("请输入编号: ").strip()

    if choice not in task_map:
        print(f"无效输入: {choice}")
        raise SystemExit(1)

    task = task_map[choice]

    if task == "Train":
        train_main()

    elif task == "Test":
        test_main()
