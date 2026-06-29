import os
import sys
import time

sys.dont_write_bytecode = True

import cv2
import joblib
import numpy as np
from skimage.feature import hog
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler

np.random.seed(42)


def extract_features(img):
    """
    从图像数组 (numpy array) 提取 HOG + HSV 颜色直方图特征

    参数:
        img: cv2 读取的图片 (BGR格式)

    返回:
        numpy.ndarray: 特征向量 (10180维)
    """
    if img is None:
        return None

    # 统一尺寸
    img = cv2.resize(img, (224, 224))

    # ---------- 提取 HOG 特征 ----------
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hog_feat = hog(
        gray,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        feature_vector=True,
    )

    # ---------- 提取 HSV 颜色直方图 ----------
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist(
        [hsv],
        [0, 1, 2],
        None,
        [16, 16, 16],
        [0, 180, 0, 256, 0, 256],
    )
    hist = cv2.normalize(hist, hist).flatten()

    # ---------- 拼接特征 ----------
    feature_vector = np.concatenate([hog_feat, hist])
    return feature_vector


def load_data(data_dir, augment=False, sliding_window=False):
    """
    遍历指定目录下的所有子文件夹，批量提取图片特征

    参数:
        data_dir: 数据集根目录
        augment: 是否启用随机裁剪增强（训练集使用）
        sliding_window: 是否启用滑动窗口采样（验证集使用）

    返回:
        X: 特征矩阵
        y: 标签列表
    """
    X, y = [], []
    if not os.path.exists(data_dir):
        print(f"警告: 目录 {data_dir} 不存在")
        return np.array(X), np.array(y)

    classes = [
        d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))
    ]

    for class_name in classes:
        folder = os.path.join(data_dir, class_name)
        file_count = 0
        for fname in os.listdir(folder):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            img_path = os.path.join(folder, fname)
            img = cv2.imread(img_path)
            if img is None:
                continue

            # 训练模式：原图 + 裁剪
            if augment:
                feat = extract_features(img)
                if feat is not None:
                    X.append(feat)
                    y.append(class_name)
                    file_count += 1
                h, w = img.shape[:2]
                for _ in range(2):
                    crop_ratio = np.random.uniform(0.40, 0.85)
                    crop_h, crop_w = int(h * crop_ratio), int(w * crop_ratio)
                    if crop_h < 10 or crop_w < 10:
                        continue
                    sx = np.random.randint(0, w - crop_w) if w > crop_w else 0
                    sy = np.random.randint(0, h - crop_h) if h > crop_h else 0
                    cropped = img[sy : sy + crop_h, sx : sx + crop_w]
                    if cropped.size == 0:
                        continue
                    feat_crop = extract_features(cropped)
                    if feat_crop is not None:
                        X.append(feat_crop)
                        y.append(class_name)
                        file_count += 1

            # 验证模式：随机采样固定数量的窗口
            elif sliding_window:
                h, w = img.shape[:2]
                scales = [0.3, 0.5, 0.7]
                num_windows = 15
                attempts = 0
                max_attempts = 100
                sampled = 0
                while sampled < num_windows and attempts < max_attempts:
                    attempts += 1
                    scale = np.random.choice(scales)
                    win_h = int(h * scale)
                    win_w = int(w * scale)
                    if win_h < 64 or win_w < 64:
                        continue
                    # 随机位置
                    sx = np.random.randint(0, w - win_w) if w > win_w else 0
                    sy = np.random.randint(0, h - win_h) if h > win_h else 0
                    window = img[sy : sy + win_h, sx : sx + win_w]
                    if window.size == 0:
                        continue
                    feat = extract_features(window)
                    if feat is not None:
                        X.append(feat)
                        y.append(class_name)
                        file_count += 1
                        sampled += 1

            # 普通模式（原图）
            else:
                feat = extract_features(img)
                if feat is not None:
                    X.append(feat)
                    y.append(class_name)
                    file_count += 1

        print(f"- 类别 '{class_name}' 成功加载 {file_count} 张图片样本")

    print(f"总计加载 {len(X)} 张图片样本")
    return np.array(X), np.array(y)


def main():
    """训练主流程"""
    # ---------- 配置路径 ----------
    train_dir = "data/train"
    valid_dir = "data/valid"
    model_dir = "models"

    # ---------- 加载训练集特征 ----------
    print("\n" + "=" * 50)
    print("加载训练集并提取特征...")
    X_train, y_train = load_data(train_dir, augment=True, sliding_window=False)
    if len(X_train) == 0:
        print("错误: 训练集为空，请检查 data/train 目录下是否有图片")
        sys.exit()

    # ---------- 加载验证集特征 ----------
    print("\n" + "=" * 50)
    print("加载验证集并提取特征...")
    X_valid, y_valid = load_data(valid_dir, augment=False, sliding_window=True)
    if len(X_valid) == 0:
        print("警告: 验证集为空，模型将无法评估泛化能力")
        sys.exit()

    # ---------- 标签编码 ----------
    print("\n" + "=" * 50)
    print("标签编码...")
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_valid_enc = le.transform(y_valid)

    print("类别与数字映射关系:")
    for idx, name in enumerate(le.classes_):
        print(f"  {idx} -> {name}")

    # ---------- 特征标准化 ----------
    print("\n" + "=" * 50)
    print("特征标准化 (Z-score)...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_valid_scaled = scaler.transform(X_valid)

    print(
        f"训练集特征均值: {X_train_scaled.mean():.4f}, 标准差: {X_train_scaled.std():.4f}"
    )
    print(
        f"验证集特征均值: {X_valid_scaled.mean():.4f}, 标准差: {X_valid_scaled.std():.4f}"
    )

    # ---------- PCA 降维 ----------
    print("\n" + "=" * 50)
    print("应用 PCA 降维...")
    pca = PCA(n_components=0.95, svd_solver="randomized", random_state=42)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_valid_pca = pca.transform(X_valid_scaled)

    print(f"降维前特征维度: {X_train_scaled.shape[1]}")
    print(f"降维后特征维度: {X_train_pca.shape[1]}")

    # ---------- 训练随机森林 ----------
    print("\n" + "=" * 50)
    print("训练随机森林分类器...")
    start_time = time.time()

    model = RandomForestClassifier(
        n_estimators=500,
        max_depth=6,
        min_samples_split=8,
        min_samples_leaf=4,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train_pca, y_train_enc)
    train_time = time.time() - start_time
    print(f"训练完成，耗时: {train_time:.2f} 秒")

    # ---------- 评估模型 ----------
    print("\n" + "=" * 50)
    print("模型评估...")

    train_pred = model.predict(X_train_pca)
    train_acc = accuracy_score(y_train_enc, train_pred)
    print(f"训练集准确率: {train_acc:.4f}")

    valid_pred = model.predict(X_valid_pca)
    valid_acc = accuracy_score(y_valid_enc, valid_pred)
    print(f"验证集准确率: {valid_acc:.4f}")

    print("\n各类别详细分类报告:")
    print(classification_report(y_valid_enc, valid_pred, target_names=le.classes_))

    # ---------- 保存模型 ----------
    print("\n" + "=" * 50)
    print("保存模型...")
    os.makedirs(model_dir, exist_ok=True)

    joblib.dump(model, os.path.join(model_dir, "landmark_model.pkl"))
    joblib.dump(scaler, os.path.join(model_dir, "scaler.pkl"))
    joblib.dump(le, os.path.join(model_dir, "label_encoder.pkl"))
    joblib.dump(pca, os.path.join(model_dir, "pca.pkl"))

    print(f"模型已成功保存至 {model_dir}/ 目录")


if __name__ == "__main__":
    main()
