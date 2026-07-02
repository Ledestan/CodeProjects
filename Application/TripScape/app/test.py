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
    """
    sift = cv2.SIFT_create()
    kp, des = sift.detectAndCompute(img, None)
    if des is None or len(des) == 0:
        return np.zeros(kmeans_model.cluster_centers_.shape[0] * 128, dtype=np.float32)
    if len(des) > max_kp:
        indices = np.random.choice(len(des), max_kp, replace=False)
        des = des[indices]
    labels, _ = vq(des, kmeans_model.cluster_centers_)
    k = kmeans_model.n_clusters
    d = des.shape[1]
    vlad = np.zeros((k, d), dtype=np.float32)
    residuals = des - kmeans_model.cluster_centers_[labels]
    np.add.at(vlad, labels, residuals)
    vlad = vlad.flatten()
    vlad = vlad / (np.linalg.norm(vlad) + 1e-8)
    return vlad


def extract_spm_hog(img):
    """
    提取 HOG 特征，输入图像先经 Letterbox 等比例缩放+填充到 256x256。
    """
    target_size = 256
    h, w = img.shape[:2]
    scale = target_size / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)
    new_w = max(1, new_w)
    new_h = max(1, new_h)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.full((target_size, target_size, 3), 0, dtype=np.uint8)
    dx = (target_size - new_w) // 2
    dy = (target_size - new_h) // 2
    canvas[dy : dy + new_h, dx : dx + new_w] = resized
    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
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

    # 二值化获取前景区域，阈值 30 可过滤掉极暗的噪声像素
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
    在 Lab 颜色空间计算颜色矩。
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    moments = []
    for ch in range(3):
        channel = lab[:, :, ch].astype(np.float32)
        mean = np.mean(channel)
        std = np.std(channel)
        skew = np.mean(((channel - mean) ** 3)) / ((std + 1e-8) ** 3)
        moments.extend([mean, std, skew])
    return np.array(moments, dtype=np.float32)


def extract_glcm(img):
    """
    基于灰度共生矩阵计算四个纹理统计量。
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (128, 128), interpolation=cv2.INTER_AREA)
    glcm = graycomatrix(
        gray, distances=[1], angles=[0, np.pi / 4], levels=256, symmetric=True
    )
    props = ["energy", "contrast", "homogeneity", "correlation"]
    features = []
    for prop in props:
        feat = graycoprops(glcm, prop).mean()
        features.append(feat)
    return np.array(features, dtype=np.float32)


def extract_features(img, kmeans_model, vlad_max_kp=100):
    """
    从单张图像提取完整的特征向量。
    现在特征长度为：VLAD(32*128=4096) + HOG(34596) + 几何(3) + 颜色矩(9) + GLCM(4) = 38708 维
    """
    vlad = extract_sift_vlad(img, kmeans_model, max_kp=vlad_max_kp)
    hog = extract_spm_hog(img)
    geom = extract_geometry(img)  # 现在返回 3 维
    color = extract_color_moments(img)
    glcm = extract_glcm(img)
    feat = np.concatenate([vlad, hog, geom, color, glcm])
    return feat.astype(np.float32)


def generate_multi_scale_windows(img, scale_ratios=None, max_windows_total=50):
    """
    根据原图短边生成多种尺度的随机滑动窗口，采样数量与尺度平方成反比。
    当某尺度可采样的位置总数少于分配数时，自动切换为密集采样。
    """
    if scale_ratios is None:
        scale_ratios = [1.0, 0.7, 0.5, 0.3]

    h, w = img.shape[:2]
    short_side = min(h, w)
    all_windows = []

    # 按面积反比计算权重
    weights = [1.0 / (r * r) for r in scale_ratios]
    total_weight = sum(weights)

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
            for i in sorted(range(len(weights)), key=lambda i: weights[i]):
                if samples_per_scale[i] > 1:
                    samples_per_scale[i] -= 1
                    break

    for ratio, n_samples in zip(scale_ratios, samples_per_scale):
        win_size = int(short_side * ratio)
        win_size = max(win_size, 64)
        if win_size > h or win_size > w:
            continue

        max_x = w - win_size
        max_y = h - win_size

        if max_x == 0 and max_y == 0:
            all_windows.append(img)
            continue

        total_positions = (max_x + 1) * (max_y + 1)
        actual_samples = min(n_samples, total_positions)

        if actual_samples == total_positions:
            for y in range(max_y + 1):
                for x in range(max_x + 1):
                    win = img[y : y + win_size, x : x + win_size]
                    all_windows.append(win)
        else:
            for _ in range(actual_samples):
                x = np.random.randint(0, max_x + 1)
                y = np.random.randint(0, max_y + 1)
                win = img[y : y + win_size, x : x + win_size]
                all_windows.append(win)

    return all_windows


class LandmarkPredictor:
    """
    地标识别预测器，默认启用多尺度贝叶斯滑动窗口。
    """

    def __init__(self, model_dir="./models", vlad_max_kp=500):
        self.model_dir = model_dir
        self.vlad_max_kp = vlad_max_kp
        print("正在加载模型...")
        self.encoder = self._load_pickle("label_encoder.pkl")
        self.kmeans = self._load_pickle("kmeans_vlad.pkl")
        self.scaler = self._load_pickle("scaler.pkl")
        self.svm = self._load_pickle("svm_model.pkl")
        print("模型加载完成。")

    def _load_pickle(self, filename):
        path = os.path.join(self.model_dir, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"模型文件不存在: {path}")
        with open(path, "rb") as f:
            return pickle.load(f)

    def predict(
        self,
        img_path,
        top_k=3,
        use_bayes=True,
        scale_ratios=None,
        max_windows_total=50,
    ):
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"无法读取图片: {img_path}")

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
        windows = generate_multi_scale_windows(
            img, scale_ratios=scale_ratios, max_windows_total=max_windows_total
        )

        if not windows:
            return self.predict(img_path, top_k, use_bayes=False)

        # 按窗口尺寸分组
        scale_groups = {}
        for win in windows:
            s = win.shape[0]
            key = round(s / 10) * 10
            scale_groups.setdefault(key, []).append(win)

        # 定义不同尺度的权重（强调大尺度全局轮廓）
        # 权重顺序与 scale_ratios 对应：[1.0, 0.7, 0.5, 0.3]
        scale_weights = {1.0: 1.2, 0.7: 1.2, 0.5: 0.8, 0.3: 0.8}
        # 若用户自定义了 scale_ratios，则自动生成权重（按比例）
        if scale_ratios is not None:
            sorted_ratios = sorted(scale_ratios, reverse=True)
            max_ratio = sorted_ratios[0]
            scale_weights = {r: (r / max_ratio) for r in scale_ratios}

        log_evidences = []
        for group_key, group_windows in scale_groups.items():
            # 估算该组对应的尺度（取组内窗口尺寸 / 短边）
            ratio_est = group_key / min(img.shape[:2])
            # 找到最近的尺度
            closest_ratio = min(scale_weights.keys(), key=lambda r: abs(r - ratio_est))
            weight = scale_weights.get(closest_ratio, 1.0)

            group_prob_sum = None
            for win in group_windows:
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
            group_prob_avg = group_prob_sum / len(group_windows)
            group_prob_avg = np.clip(group_prob_avg, 1e-12, 1.0)
            # 加入尺度权重
            log_evidences.append(np.log(group_prob_avg) * weight)

        total_log = np.sum(log_evidences, axis=0)
        max_log = np.max(total_log)
        exp_log = np.exp(total_log - max_log)
        final_probas = exp_log / np.sum(exp_log)

        top_indices = np.argsort(final_probas)[::-1][:top_k]
        results = [
            (self.encoder.classes_[idx], final_probas[idx]) for idx in top_indices
        ]
        return results


if __name__ == "__main__":
    # 用法: python test.py <图片路径> [top_k] [--no-bayes] [--windows N]
    if len(sys.argv) < 2:
        print("用法: python app/test.py <图片路径> [top_k] [--no-bayes] [--windows N]")
        print("默认启用贝叶斯滑动窗口，禁用: --no-bayes")
        print("自定义窗口数: --windows N")
        sys.exit(1)

    img_path = sys.argv[1]
    top_k = 3
    use_bayes = True
    max_windows = 50

    i = 2
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg.isdigit():
            top_k = int(arg)
        elif arg == "--no-bayes":
            use_bayes = False
        elif arg == "--windows" and i + 1 < len(sys.argv):
            max_windows = int(sys.argv[i + 1])
            i += 1
        i += 1

    predictor = LandmarkPredictor(model_dir="./models", vlad_max_kp=500)

    start_time = time.time()
    results = predictor.predict(
        img_path, top_k=top_k, use_bayes=use_bayes, max_windows_total=max_windows
    )
    elapsed = time.time() - start_time

    print(f"\n预测结果 (Top-{top_k})：")
    for label, conf in results:
        print(f"- {label}: {conf:.4f}")
    print(f"模式: {'贝叶斯多尺度滑动窗口' if use_bayes else '整图'}")
    if use_bayes:
        print(f"窗口总数: {max_windows}")
    print(f"推理耗时: {elapsed:.3f} 秒")
