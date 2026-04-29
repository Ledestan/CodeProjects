"""
项目名称: 图像识别分类器
创建日期: 2026-04-22
"""

import os
import numpy as np
from PIL import Image


class ImageClassifier:
    """
    核心步骤:
        Step1: 用二重积分将 224*224 灰度图转化为 3 个归一化特征
        Step2: 构建 Sigmoid 非线性模型: y = 1/(1+e^{-(w·x+b)})
        Step3: 定义交叉熵损失函数
        Step4: 计算损失对参数的梯度
        Step5: 梯度下降迭代训练
        Step6: 模型测试与单张预测
    """

    def __init__(self):
        """初始化分类器，预计算归一化常量和坐标网格"""
        self.img_size = (224, 224)
        self.H, self.W = 224, 224
        
        # 特征归一化理论最大值 (全白图片)
        self.m1 = self.H * self.W * 255.0
        self.m23 = self.H * self.W * 255.0 * self.W
        
        # 预计算坐标网格
        self.x_coords = np.arange(1, self.W + 1, dtype=np.float64).reshape(1, -1)
        self.y_coords = np.arange(1, self.H + 1, dtype=np.float64).reshape(-1, 1)
        
        # 模型参数 (待训练)
        self.w = None # shape (3,)
        self.b = None # 标量
        
        # 训练历史记录
        self.loss_history = []
    
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
        for class_name, label in [('cat', 1), ('dog', 0)]:
            # 访问文件夹文件, 检查格式
            class_dir = os.path.join(data_dir, class_name)
            if not os.path.isdir(class_dir):
                print(f"警告: 目录 {class_dir} 不存在，跳过")
                continue
            for fname in sorted(os.listdir(class_dir)):
                if fname.lower().endswith((".jpg", ".png", ".jpeg")):
                    img_path = os.path.join(class_dir, fname)
                    img = self.load_single_image(img_path)
                    # 确认读取成功保存
                    if img is not None:
                        images.append(img)
                        labels.append(label)
        return images, labels
    
    def load_single_image(self, path: str):
        """加载单张图片，转为灰度，缩放至指定尺寸"""
        try:
            img = Image.open(path).convert('L') # 灰度
            img = img.resize(self.img_size, Image.Resampling.LANCZOS)
            return np.array(img, dtype=np.float64)
        except Exception as e:
            print(f"无法加载图片 '{path}': {e}")
            return None
    
    # ==================== Step1: 特征提取 (二重积分 + 归一化) ====================
    def extract_features(self, img):
        """
        对单张灰度图片计算三个归一化特征

        特征定义 (离散二重积分):
            x1_hat = ΣΣ f(x,y)      -> 灰度总量
            x2_hat = ΣΣ x·f(x,y)    -> 横向灰度加权
            x3_hat = ΣΣ y·f(x,y)    -> 纵向灰度加权
        归一化:
            x1 = x1_hat / m1
            x2 = x2_hat / m23
            x3 = x3_hat / m23
        
        参数:
            img: np.ndarray, shape (H, W), 灰度值 0~255
        
        返回:
            features: np.ndarray, shape (3,), [x1, x2, x3]
        """
        x1_hat = np.sum(img)
        x2_hat = np.sum(img * self.x_coords)
        x3_hat = np.sum(img * self.y_coords)
        
        x1 = x1_hat / self.m1
        x2 = x2_hat / self.m23
        x3 = x3_hat / self.m23
        
        return np.array([x1, x2, x3])
    
    def prepare_dataset(self, images):
        """
        将图片列表转换为特征矩阵
        
        参数:
            images: list of np.ndarray
        
        返回:
            features: np.ndarray, shape (N, 3)
        """
        features = [self.extract_features(img) for img in images]
        return np.array(features)
    
    # ==================== Step2: 非线性模型 (Sigmoid) ====================
    @staticmethod
    def sigmoid(t):
        """
        Sigmoid 激活函数: y = 1 / (1 + e^{-t})

        使用数值裁剪防止溢出
        """
        t = np.clip(t, -500, 500) # exp(500) 接近浮点数上限
        return 1.0 / (1.0 + np.exp(-t))
    
    def forward(self, features):
        """
        前向传播: 计算预测概率
        
        公式: t = w·x + b,  y = sigmoid(t)
        
        参数:
            features: np.ndarray, shape (N, 3) 或 (3,)
        
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
            features: np.ndarray, shape (N, 3)
            y_true: np.ndarray, shape (N,)
        
        返回:
            dw: np.ndarray, shape (3,)
            db: float
        """
        y_pred = self.forward(features)
        error = y_pred - y_true # (N,)
        
        dw = np.mean(error[:, np.newaxis] * features, axis=0) # (3,)
        db = np.mean(error)
        return dw, db
    
    # ==================== Step5: 梯度下降训练 ====================
    def train(self, features, labels, learning_rate=0.01, tolerance=1e-5, max_iters=10000, verbose=True):
        """
        使用梯度下降训练模型参数
        
        参数:
            features: np.ndarray, shape (N, 3)
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
        # 参数初始化 (小随机数)
        np.random.seed(0)
        self.w = np.random.randn(3) * 0.01
        self.b = 0.0
        
        self.loss_history = []
        prev_loss = float('inf')
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
            if verbose and (i % 200 == 0 or i < 10):
                print(f"Iter {i:4d} | Loss: {loss:.6f} | w: [{self.w[0]:.4f}, {self.w[1]:.4f}, {self.w[2]:.4f}], b: {self.b:.4f}")
        
        if verbose:
            print(f"\n训练完成！")
            print(f"最优参数: w = [{self.w[0]:.6f}, {self.w[1]:.6f}, {self.w[2]:.6f}], b = {self.b:.6f}")
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
        
        test_imgs, test_labels = self.load_images(test_dir)
        if not test_imgs:
            return 0.0, []
        
        test_features = self.prepare_dataset(test_imgs)
        probs = self.forward(test_features)     # 预测概率
        preds = (probs >= 0.5).astype(int)      # 阈值 0.5 判定类别
        
        correct = np.sum(preds == test_labels)
        accuracy = correct / len(test_labels)
        
        results = []
        for i, (prob, pred, true_label) in enumerate(zip(probs, preds, test_labels)):
            results.append({
                'index': i,
                'true_label': true_label,
                'true_name': '猫' if true_label == 1 else '狗',
                'pred_prob': prob,
                'pred_label': pred,
                'pred_name': '猫' if pred == 1 else '狗',
                'correct': (pred == true_label)
            })
        
        print(f"\n测试集评估结果:")
        print(f"总样本数: {len(test_labels)}")
        print(f"正确数: {correct}")
        print(f"准确率: {accuracy:.2%}")
        
        # 打印错误详情
        errors = [r for r in results if not r['correct']]
        if errors:
            print(f"错误预测的样本 ({len(errors)} 个):")
            for err in errors:
                print(f"图片 {err['index']}: 真实={err['true_name']}, 预测概率={err['pred_prob']:.4f} → {err['pred_name']}")
        
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
            return {'success': False, 'error': '模型未初始化'}
        
        img = self.load_single_image(img_path)
        if img is None:
            return {'success': False, 'error': '无法加载图片'}
        
        features = self.extract_features(img).reshape(1, -1)
        prob = self.forward(features)[0]
        pred_label = 1 if prob >= 0.5 else 0
        pred_name = '猫' if pred_label == 1 else '狗'
        
        return {
            'success': True,
            'probability': float(prob),
            'class_name': pred_name,
            'class_label': pred_label
        }
    
    # ==================== 参数持久化 ====================
    def save_params(self, filepath):
        """保存模型参数到 .npz 文件"""
        if self.w is None or self.b is None:
            raise ValueError("没有可保存的参数，请先训练模型")
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        np.savez(filepath, w=self.w, b=self.b)
        print(f"参数已保存至: {filepath}")
    
    def load_params(self, filepath):
        """从 .npz 文件加载模型参数"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"参数文件不存在: {filepath}")
        data = np.load(filepath)
        self.w = data['w']
        self.b = data['b'].item()  # 提取标量
        print(f"参数已从 {filepath} 加载: w={self.w}, b={self.b:.6f}")