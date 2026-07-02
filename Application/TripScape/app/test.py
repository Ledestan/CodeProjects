import os
import pickle
import sys
import time
import warnings

import cv2
import numpy as np
from scipy.cluster.vq import vq
from skimage.feature import graycomatrix, graycoprops
from sklearn.exceptions import InconsistentVersionWarning

warnings.filterwarnings("ignore", category=InconsistentVersionWarning)


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


def generate_multi_scale_windows(img, scale_ratios=None, max_windows_total=100):
    """
    根据原图短边生成多种尺度的随机滑动窗口，采样数量与尺度平方成反比。
    当某尺度可采样的位置总数少于分配数时，自动切换为密集采样（取尽所有位置）。

    参数:
        img : numpy.ndarray, 原图 (BGR)
        scale_ratios : list, 窗口边长占短边的比例，默认 [1.0, 0.7, 0.5, 0.3]
        max_windows_total : int, 所有尺度生成的窗口总数上限

    返回:
        list of numpy.ndarray, 所有窗口图像（BGR）
    """
    if scale_ratios is None:
        scale_ratios = [1.0, 0.7, 0.5, 0.3]

    h, w = img.shape[:2]
    short_side = min(h, w)
    all_windows = []

    # 按面积反比计算权重，面积越小权重越大
    weights = [1.0 / (r * r) for r in scale_ratios]
    total_weight = sum(weights)

    # 按权重分配窗口数量，确保每个尺度至少1个
    samples_per_scale = []
    for wgt in weights:
        n = int(max_windows_total * (wgt / total_weight))
        n = max(1, n)
        samples_per_scale.append(n)

    # 补足或削减至总数正好为 max_windows_total
    while sum(samples_per_scale) < max_windows_total:
        max_idx = weights.index(max(weights))
        samples_per_scale[max_idx] += 1
    while sum(samples_per_scale) > max_windows_total:
        min_idx = weights.index(min(weights))
        if samples_per_scale[min_idx] > 1:
            samples_per_scale[min_idx] -= 1
        else:
            # 若最小尺度只有1个，从次小尺度削减
            for i in sorted(range(len(weights)), key=lambda i: weights[i]):
                if samples_per_scale[i] > 1:
                    samples_per_scale[i] -= 1
                    break

    # 生成窗口
    for ratio, n_samples in zip(scale_ratios, samples_per_scale):
        win_size = int(short_side * ratio)
        win_size = max(win_size, 64)  # 确保最小尺寸，避免特征提取失败
        if win_size > h or win_size > w:
            continue

        max_x = w - win_size
        max_y = h - win_size

        # 若图像极小，无法滑动，直接添加原图
        if max_x == 0 and max_y == 0:
            all_windows.append(img)
            continue

        total_positions = (max_x + 1) * (max_y + 1)
        actual_samples = min(n_samples, total_positions)

        # 如果可采位置总数少于计划采样数，进行密集采样（覆盖全部可能位置）
        if actual_samples == total_positions:
            for y in range(max_y + 1):
                for x in range(max_x + 1):
                    win = img[y : y + win_size, x : x + win_size]
                    all_windows.append(win)
        else:
            # 否则随机采样
            for _ in range(actual_samples):
                x = np.random.randint(0, max_x + 1)
                y = np.random.randint(0, max_y + 1)
                win = img[y : y + win_size, x : x + win_size]
                all_windows.append(win)

    return all_windows


class LandmarkPredictor:
    """
    地标识别预测器，支持整图推理和多尺度贝叶斯滑动窗口投票。

    用法:
        predictor = LandmarkPredictor(model_dir="./models", vlad_max_kp=500)
        results = predictor.predict("path/to/image.jpg", top_k=3)
        # results = [('Eiffel', 0.85), ('Leifeng', 0.10), ...]

    默认启用贝叶斯滑动窗口，可通过 use_bayes=False 关闭。
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

    def predict(
        self,
        img_path,
        top_k=3,
        use_bayes=True,
        scale_ratios=None,
        max_windows_total=100,
    ):
        """
        对单张图片进行地标识别，返回 Top-K 结果。

        参数:
            img_path : str, 图片文件路径
            top_k : int, 返回前 k 个最可能的类别
            use_bayes : bool, 是否启用多尺度滑动窗口贝叶斯投票，默认 True
            scale_ratios : list, 窗口边长占短边的比例，默认 [1.0, 0.7, 0.5, 0.3]
            max_windows_total : int, 贝叶斯模式下的最大窗口总数

        返回:
            list of (label, confidence)
                label : str, 地标名称
                confidence : float, 置信度（0~1）
        """
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"无法读取图片: {img_path}")

        # 整图推理模式（快速，但可能牺牲偏移/尺度鲁棒性）
        if not use_bayes:
            feat = extract_features(img, self.kmeans, vlad_max_kp=self.vlad_max_kp)
            feat = feat.reshape(1, -1)
            feat_scaled = self.scaler.transform(feat)
            if hasattr(self.svm, "predict_proba"):
                probas = self.svm.predict_proba(feat_scaled)[0]
            else:
                scores = self.svm.decision_function(feat_scaled)[0]
                exp_scores = np.exp(scores - np.max(scores))
                probas = exp_scores / np.sum(exp_scores)
            top_indices = np.argsort(probas)[::-1][:top_k]
            results = [(self.encoder.classes_[idx], probas[idx]) for idx in top_indices]
            return results

        # ---------- 多尺度贝叶斯滑动窗口 ----------
        # 生成所有尺度的随机窗口（总数受 max_windows_total 限制）
        windows = generate_multi_scale_windows(
            img, scale_ratios=scale_ratios, max_windows_total=max_windows_total
        )

        if not windows:
            # 极端情况（如图片极小无法滑动），回退整图
            return self.predict(img_path, top_k, use_bayes=False)

        # 按窗口尺寸分组（同一尺度的窗口尺寸相同）
        scale_groups = {}
        for win in windows:
            s = win.shape[0]  # 正方形窗口，宽高相等
            key = round(s / 10) * 10  # 四舍五入到最近的10像素作为组键
            scale_groups.setdefault(key, []).append(win)

        log_evidences = []
        for group_key, group_windows in scale_groups.items():
            group_prob_sum = None
            for win in group_windows:
                # 提取窗口特征
                feat = extract_features(win, self.kmeans, vlad_max_kp=self.vlad_max_kp)
                feat = feat.reshape(1, -1)
                feat_scaled = self.scaler.transform(feat)
                if hasattr(self.svm, "predict_proba"):
                    prob = self.svm.predict_proba(feat_scaled)[0]
                else:
                    scores = self.svm.decision_function(feat_scaled)[0]
                    exp_scores = np.exp(scores - np.max(scores))
                    prob = exp_scores / np.sum(exp_scores)
                if group_prob_sum is None:
                    group_prob_sum = prob
                else:
                    group_prob_sum += prob
            # 组内概率取平均，作为该尺度的证据
            group_prob_avg = group_prob_sum / len(group_windows)
            # 截断防止 log 取到负无穷
            group_prob_avg = np.clip(group_prob_avg, 1e-12, 1.0)
            log_evidences.append(np.log(group_prob_avg))

        # 跨尺度累加 log 证据（每个尺度等权）
        total_log = np.sum(log_evidences, axis=0)
        # 归一化为概率分布（减去最大值防溢出）
        max_log = np.max(total_log)
        exp_log = np.exp(total_log - max_log)
        final_probas = exp_log / np.sum(exp_log)

        top_indices = np.argsort(final_probas)[::-1][:top_k]
        results = [
            (self.encoder.classes_[idx], final_probas[idx]) for idx in top_indices
        ]
        return results


if __name__ == "__main__":
    # 用法: python test.py <图片路径> [top_k] [--no-bayes]
    # 默认启用贝叶斯滑动窗口，禁用添加 --no-bayes 参数
    if len(sys.argv) < 2:
        print("用法: python app/test.py <图片路径> [top_k] [--no-bayes]")
        print("默认启用贝叶斯滑动窗口，禁用添加 --no-bayes 参数")
        sys.exit(1)

    img_path = sys.argv[1]
    top_k = 3
    use_bayes = True

    # 解析命令行参数
    for arg in sys.argv[2:]:
        if arg.isdigit():
            top_k = int(arg)
        elif arg == "--no-bayes":
            use_bayes = False

    # 创建预测器实例，vlad_max_kp 可在此调整（200~500）
    predictor = LandmarkPredictor(model_dir="./models", vlad_max_kp=500)

    start_time = time.time()
    results = predictor.predict(img_path, top_k=top_k, use_bayes=use_bayes)
    elapsed = time.time() - start_time

    print(f"\n预测结果 (Top-{top_k})：")
    for label, conf in results:
        print(f"- {label}: {conf:.4f}")
    print(f"模式: {'贝叶斯多尺度滑动窗口' if use_bayes else '整图'}")
    print(f"推理耗时: {elapsed:.3f} 秒")
