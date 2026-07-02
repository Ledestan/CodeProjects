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


def extract_sift_vlad(img, kmeans_model, max_kp):
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


def extract_spm_hog(img, target_size):
    """
    提取 HOG 特征，输入图像先经 Letterbox 等比例缩放+填充到 target_size x target_size。
    """
    h, w = img.shape[:2]
    scale = target_size / max(h, w)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
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


def extract_profile(img, segments):
    """
    提取轴向宽度轮廓曲线特征，用于刻画建筑（尤其是塔类）的纵向形状变化。

    该特征将二值化前景沿垂直方向均匀分为多个水平段，统计每段的最大宽度，形成轮廓曲线。
    不同建筑具有独特曲线模式：金字塔呈线性递增，天坛呈蘑菇状（顶部突宽），
    崇圣寺塔呈平滑弧线收分，雷峰塔近似矩形带飞檐，埃菲尔铁塔呈指数级下宽上窄。

    参数:
        img : numpy.ndarray, BGR 图像
        segments : int, 垂直分段数

    返回:
        numpy.ndarray, 长度为 segments 的归一化宽度曲线，数值范围 [0, 1]
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
    h, w = thresh.shape
    split_points = np.linspace(0, h, segments + 1, dtype=int)
    widths = []
    for i in range(segments):
        roi = thresh[split_points[i] : split_points[i + 1], :]
        if roi.size == 0:
            widths.append(0)
            continue
        row_sums = np.sum(roi > 0, axis=1)
        if np.max(row_sums) == 0:
            widths.append(0)
        else:
            widths.append(np.max(row_sums))
    widths = np.array(widths, dtype=np.float32)
    if np.max(widths) > 0:
        widths = widths / np.max(widths)
    return widths


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


def extract_raw_features(img, kmeans_model, vlad_max_kp, hog_target_size, profile_segments):
    """
    提取原始特征（含完整 HOG），顺序：[VLAD, HOG, 轮廓, 颜色矩, GLCM]
    与训练代码中的 extract_features 完全一致。
    """
    vlad = extract_sift_vlad(img, kmeans_model, vlad_max_kp)
    hog = extract_spm_hog(img, hog_target_size)
    profile = extract_profile(img, profile_segments)
    color = extract_color_moments(img)
    glcm = extract_glcm(img)
    feat = np.concatenate([vlad, hog, profile, color, glcm])
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

    def __init__(self, model_dir="./models"):
        self.model_dir = model_dir
        # 特征提取参数（与训练时保持一致）
        self.vlad_max_kp = 500
        self.hog_target_size = 256
        self.profile_segments = 20

        print("正在加载模型...")
        self.encoder = self._load_pickle("label_encoder.pkl")
        self.kmeans = self._load_pickle("kmeans_vlad.pkl")
        self.pca = self._load_pickle("pca_hog.pkl")       # PCA 降维模型
        self.scaler = self._load_pickle("scaler.pkl")
        self.svm = self._load_pickle("svm_model.pkl")     # 训练保存的线性SVM（已改为 svm_model.pkl）

        # 从模型中获取维度信息，用于特征切片
        self.vlad_dim = self.kmeans.n_clusters * 128
        self.hog_dim = self.pca.n_features_in_           # PCA 训练时的输入维度（原始HOG长度）
        print("模型加载完成。")

    def _load_pickle(self, filename):
        path = os.path.join(self.model_dir, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"模型文件不存在: {path}")
        with open(path, "rb") as f:
            return pickle.load(f)

    def _transform_features(self, raw_feat):
        """
        对原始特征进行 PCA 降维并拼接，得到最终特征（与训练时完全一致）。
        raw_feat: 一维 numpy 数组，顺序为 [VLAD, HOG, 轮廓, 颜色, GLCM]
        返回: 一维 numpy 数组，顺序为 [VLAD, PCA-HOG, 轮廓, 颜色, GLCM]
        """
        # 切片
        vlad = raw_feat[:self.vlad_dim]
        hog_start = self.vlad_dim
        hog_end = hog_start + self.hog_dim
        hog_raw = raw_feat[hog_start:hog_end]
        profile_start = hog_end
        profile_end = profile_start + self.profile_segments
        color_start = profile_end
        color_end = color_start + 9
        glcm_start = color_end
        glcm_end = glcm_start + 4

        profile = raw_feat[profile_start:profile_end]
        color = raw_feat[color_start:color_end]
        glcm = raw_feat[glcm_start:glcm_end]

        # PCA 降维 HOG
        hog_pca = self.pca.transform(hog_raw.reshape(1, -1)).flatten()

        # 重新拼接
        final_feat = np.concatenate([vlad, hog_pca, profile, color, glcm])
        return final_feat

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
            # 整图预测
            raw = extract_raw_features(
                img, self.kmeans, self.vlad_max_kp,
                self.hog_target_size, self.profile_segments
            )
            final = self._transform_features(raw)
            feat_scaled = self.scaler.transform(final.reshape(1, -1))
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

        # 尺度权重（按比例）
        if scale_ratios is None:
            scale_ratios = [1.0, 0.7, 0.5, 0.3]
        sorted_ratios = sorted(scale_ratios, reverse=True)
        max_ratio = sorted_ratios[0]
        scale_weights = {r: (r / max_ratio) for r in scale_ratios}

        log_evidences = []
        for group_key, group_windows in scale_groups.items():
            ratio_est = group_key / min(img.shape[:2])
            closest_ratio = min(scale_weights.keys(), key=lambda r: abs(r - ratio_est))
            weight = scale_weights.get(closest_ratio, 1.0)

            group_prob_sum = None
            for win in group_windows:
                raw = extract_raw_features(
                    win, self.kmeans, self.vlad_max_kp,
                    self.hog_target_size, self.profile_segments
                )
                final = self._transform_features(raw)
                feat_scaled = self.scaler.transform(final.reshape(1, -1))
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
    # 用法: python app/test.py <图片路径> [top_k] [--no-bayes] [--windows N]
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

    predictor = LandmarkPredictor(model_dir="./models")

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