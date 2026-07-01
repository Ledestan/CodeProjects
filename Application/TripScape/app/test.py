import os
import pickle
import sys
import time

import cv2
import numpy as np
from scipy.cluster.vq import vq
from skimage.feature import graycomatrix, graycoprops


def extract_sift_vlad(img, kmeans_model, max_kp=100):
    """
    提取 SIFT 关键点并计算 VLAD 向量。

    SIFT 检测图像中的尺度不变关键点，生成 128 维描述子。
    VLAD 通过将每个描述子分配到最近的聚类中心（码本），
    并累加描述子与中心的残差，得到固定长度的全局图像表示，
    相比词袋模型保留了更多的空间分布信息。

    参数:
        img : numpy.ndarray, BGR 图像
        kmeans_model : 已训练的 KMeans 聚类器（码本）
        max_kp : int, 单张图片最多参与 VLAD 编码的关键点数量（随机采样）
                 用于控制推理速度，降低计算量

    返回:
        numpy.ndarray, VLAD 特征向量，维度 = n_clusters × 128
        若无关键点则返回零向量，保证特征维度一致。
    """
    sift = cv2.SIFT_create()
    kp, des = sift.detectAndCompute(img, None)

    # 若图片无关键点（如纯色背景），直接返回零向量，避免后续报错
    if des is None or len(des) == 0:
        return np.zeros(kmeans_model.cluster_centers_.shape[0] * 128, dtype=np.float32)

    # 随机采样关键点，控制参与编码的数量，显著加速推理
    # 采样保留特征的统计分布特性，对精度影响较小
    if len(des) > max_kp:
        indices = np.random.choice(len(des), max_kp, replace=False)
        des = des[indices]

    # 硬分配：将每个描述子分配到最近的聚类中心
    labels, _ = vq(des, kmeans_model.cluster_centers_)
    k = kmeans_model.n_clusters  # 聚类数量，例如64
    d = des.shape[1]  # SIFT维度固定为128

    # 初始化 VLAD 矩阵，形状 (k, d)
    vlad = np.zeros((k, d), dtype=np.float32)

    # 累加残差：描述子向量减去其对应聚类中心
    # 使用 np.add.at 实现向量化累加，避免 Python 循环，提升速度
    residuals = des - kmeans_model.cluster_centers_[labels]
    np.add.at(vlad, labels, residuals)

    # 展平并做 L2 归一化，使得特征向量的模长为1，
    # 降低光照变化和图像对比度的影响
    vlad = vlad.flatten()
    vlad = vlad / (np.linalg.norm(vlad) + 1e-8)
    return vlad


def extract_spm_hog(img):
    """
    提取 HOG（方向梯度直方图）特征，输入图像先经 Letterbox 等比例缩放+填充。

    HOG 统计图像局部区域的梯度方向分布，描述物体的边缘和轮廓信息，
    对光照变化和小的平移具有鲁棒性。
    Letterbox 处理将长边缩放到 256，短边等比例缩放，并用黑色填充剩余区域，
    避免直接拉伸导致物体变形。黑色区域的梯度为零，不会干扰 HOG 统计。

    参数:
        img : numpy.ndarray, BGR 图像

    返回:
        numpy.ndarray, HOG 特征向量（约 3780 维）
    """
    target_size = 256
    h, w = img.shape[:2]

    # 计算等比例缩放后的新尺寸，长边缩放到 target_size
    scale = target_size / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)
    new_w = max(1, new_w)
    new_h = max(1, new_h)

    # 使用 INTER_AREA 插值，适合图像缩小，减少锯齿效应
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # 创建黑色画布（填充值为0），梯度为0，不影响HOG统计
    canvas = np.full((target_size, target_size, 3), 0, dtype=np.uint8)
    dx = (target_size - new_w) // 2
    dy = (target_size - new_h) // 2
    canvas[dy : dy + new_h, dx : dx + new_w] = resized

    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)

    # 创建 HOG 描述子，使用与训练时完全相同的参数
    hog = cv2.HOGDescriptor(
        _winSize=(target_size, target_size),
        _blockSize=(16, 16),
        _blockStride=(8, 8),
        _cellSize=(8, 8),
        _nbins=9,
    )
    hog_feat = hog.compute(gray).flatten()
    return hog_feat


def extract_geometry(img):
    """
    提取两个几何形状描述子：长宽比和凸包面积比。

    长宽比（高度/宽度）用于区分高塔（细长）与宽体建筑（扁平）。
    凸包面积比（轮廓面积 / 凸包面积）反映形状的紧凑程度，
    规整建筑物（如金字塔）该值接近1，而不规则自然物（如山脉）则较小。

    参数:
        img : numpy.ndarray, BGR 图像

    返回:
        numpy.ndarray, 长度为2的数组 [aspect, area_ratio]
    """
    h, w = img.shape[:2]
    aspect = h / (w + 1e-6)  # 避免除以零

    # 二值化提取前景区域，阈值 30 可过滤掉极暗的噪声像素
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 如果没有轮廓，则无法计算面积比，直接返回默认值
    if not contours:
        return np.array([aspect, 1.0], dtype=np.float32)

    # 取面积最大的轮廓作为图像主体（假设地标占主要部分）
    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    area_ratio = area / (hull_area + 1e-6)
    return np.array([aspect, area_ratio], dtype=np.float32)


def extract_color_moments(img):
    """
    在 Lab 颜色空间计算颜色矩：一阶矩（均值）、二阶矩（标准差）、三阶矩（偏度）。

    颜色矩是一种非常紧凑的颜色特征，每个通道仅需三个值，总维度 9。
    Lab 空间更接近人类视觉感知，且对光照变化有一定鲁棒性。
    偏度描述灰度分布的对称性，有助于区分颜色分布差异大的场景。

    参数:
        img : numpy.ndarray, BGR 图像

    返回:
        numpy.ndarray, 长度为9的特征向量
            [L_mean, L_std, L_skew, a_mean, a_std, a_skew, b_mean, b_std, b_skew]
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    moments = []
    for ch in range(3):
        channel = lab[:, :, ch].astype(np.float32)
        mean = np.mean(channel)
        std = np.std(channel)
        # 偏度：E[(X-μ)^3] / σ^3，值正表示右偏，负表示左偏
        skew = np.mean(((channel - mean) ** 3)) / ((std + 1e-8) ** 3)
        moments.extend([mean, std, skew])
    return np.array(moments, dtype=np.float32)


def extract_glcm(img):
    """
    基于灰度共生矩阵（GLCM）计算四个纹理统计量：能量、对比度、同质性、相关性。

    GLCM 统计图像中一定距离和方向上的像素对灰度级联合分布，
    从而描述纹理的粗细、方向性和规律性。
    本实现同时使用 0° 和 45° 两个方向，并取平均值以获得一定旋转不变性。
    先将灰度图缩放至 128×128，以平衡计算速度和纹理稳定性。

    参数:
        img : numpy.ndarray, BGR 图像

    返回:
        numpy.ndarray, 长度为4的特征向量
            [energy, contrast, homogeneity, correlation]
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 缩放至固定尺寸，使用 INTER_AREA 保留纹理分布细节
    gray = cv2.resize(gray, (128, 128), interpolation=cv2.INTER_AREA)

    # 计算 GLCM：距离为1，角度 0° 和 45°，灰度级256，对称矩阵
    glcm = graycomatrix(
        gray, distances=[1], angles=[0, np.pi / 4], levels=256, symmetric=True
    )

    props = ["energy", "contrast", "homogeneity", "correlation"]
    features = []
    for prop in props:
        # 对两个方向的统计值取平均，降低方向敏感性
        feat = graycoprops(glcm, prop).mean()
        features.append(feat)
    return np.array(features, dtype=np.float32)


def extract_features(img, kmeans_model, vlad_max_kp=100):
    """
    从单张图片（已读入的 numpy 数组）提取完整的特征向量。

    特征组成顺序：[VLAD, HOG, 几何, 颜色矩, GLCM]。
    此函数汇总所有子特征，确保与训练时的顺序和拼接方式完全一致。

    参数:
        img : numpy.ndarray, BGR 图像
        kmeans_model : 已训练的 VLAD 码本
        vlad_max_kp : int, VLAD 编码时每张图使用的最大关键点数

    返回:
        numpy.ndarray, 拼接后的完整特征向量（约 38700 维）
    """
    # 支路A：结构特征（适合建筑、塔等刚性地标）
    vlad = extract_sift_vlad(img, kmeans_model, max_kp=vlad_max_kp)
    hog = extract_spm_hog(img)
    geom = extract_geometry(img)

    # 支路B：颜色与纹理特征（适合自然景观）
    color = extract_color_moments(img)
    glcm = extract_glcm(img)

    # 按固定顺序拼接，确保特征维度与训练时匹配
    feat = np.concatenate([vlad, hog, geom, color, glcm])
    return feat.astype(np.float32)


class LandmarkPredictor:
    """
    地标识别预测器。

    用法:
        predictor = LandmarkPredictor(model_dir="./models", vlad_max_kp=100)
        results = predictor.predict("path/to/image.jpg", top_k=3)
        # results = [('Eiffel', 0.85), ('Leifeng', 0.10), ...]

    内部会加载训练阶段保存的 label_encoder, kmeans_vlad, scaler, svm_model，
    并确保特征提取参数与训练时一致。
    """

    def __init__(self, model_dir="./models", vlad_max_kp=500):
        """
        初始化预测器，加载所有模型文件。

        参数:
            model_dir : str, 存放模型的目录（包含四个 .pkl 文件）
            vlad_max_kp : int, VLAD 编码时每张图使用的最大关键点数，
                          影响推理速度，可调。
        """
        self.model_dir = model_dir
        self.vlad_max_kp = vlad_max_kp

        # 加载模型文件，如果缺失则抛出异常，提示用户先运行 train.py
        print("正在加载模型...")
        self.encoder = self._load_pickle("label_encoder.pkl")
        self.kmeans = self._load_pickle("kmeans_vlad.pkl")
        self.scaler = self._load_pickle("scaler.pkl")
        self.svm = self._load_pickle("svm_model.pkl")
        print("模型加载完成。")

    def _load_pickle(self, filename):
        """
        内部辅助函数：从 model_dir 加载 pickle 文件。
        """
        path = os.path.join(self.model_dir, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"模型文件不存在: {path}。请先运行 train.py 生成模型。"
            )
        with open(path, "rb") as f:
            return pickle.load(f)

    def predict(self, img_path, top_k=3):
        """
        对单张图片进行地标识别，返回 Top-K 结果。

        参数:
            img_path : str, 图片文件路径
            top_k : int, 返回前 k 个最可能的类别

        返回:
            list of (label, confidence)
                label : str, 地标名称（如 'Eiffel'）
                confidence : float, 置信度（0~1），
                             来自 SVM 的概率估计或决策函数映射。
        """
        # 读取图片，若失败则抛出异常
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"无法读取图片: {img_path}")

        # 提取特征，使用与训练时相同的 max_kp 参数
        feat = extract_features(img, self.kmeans, vlad_max_kp=self.vlad_max_kp)
        feat = feat.reshape(1, -1)  # 转为行向量

        # 标准化：使用训练时拟合的 scaler，保证分布对齐
        feat_scaled = self.scaler.transform(feat)

        # 获取分类置信度：
        # 若 SVM 训练时开启了 probability=True，则使用 predict_proba 获得真实概率。
        # 否则，使用 decision_function 的得分并通过 softmax 映射到 [0,1]，
        # 作为近似置信度（虽然不严格是概率，但排序效果一致）。
        if hasattr(self.svm, "predict_proba"):
            probas = self.svm.predict_proba(feat_scaled)[0]
        else:
            scores = self.svm.decision_function(feat_scaled)[0]
            # 通过减去最大值再 exp，避免数值溢出，然后归一化得到软概率
            exp_scores = np.exp(scores - np.max(scores))
            probas = exp_scores / np.sum(exp_scores)

        # 按概率降序排序，取前 top_k 个索引
        top_indices = np.argsort(probas)[::-1][:top_k]
        results = []
        for idx in top_indices:
            label = self.encoder.classes_[idx]
            confidence = probas[idx]
            results.append((label, confidence))

        return results


if __name__ == "__main__":
    # 用法: python app/test.py <图片路径> [top_k]
    img_path = sys.argv[1]
    top_k = int(sys.argv[2]) if len(sys.argv) > 2 else 3

    # 创建预测器实例，可根据需要调整 vlad_max_kp
    predictor = LandmarkPredictor(model_dir="./models", vlad_max_kp=500)

    # 执行预测并计时
    start_time = time.time()
    results = predictor.predict(img_path, top_k=top_k)
    elapsed = time.time() - start_time

    # 打印结果
    print(f"\n预测结果 (Top-{top_k})：")
    for label, conf in results:
        print(f"- {label}: {conf:.4f}")
    print(f"\n推理耗时: {elapsed:.3f} 秒")
