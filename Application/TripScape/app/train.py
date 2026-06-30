import os
import pickle

import cv2
import numpy as np
from scipy.cluster.vq import vq
from skimage.feature import graycomatrix, graycoprops
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC


def load_data(data_dir):
    """
    从指定目录加载图片路径和对应的类别名称。

    目录结构要求：
        data_dir/类别名/*.jpg (或 .jpeg/.png)

    参数
    ----------
    data_dir : str
        数据集根目录路径。

    返回
    -------
    image_paths : list of str
        所有图片的完整路径列表。
    label_names : list of str
        与路径一一对应的类别名称（原始字符串）列表。
    """
    image_paths = []
    label_names = []
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"目录不存在: {data_dir}")
    # 获取所有子目录作为类别，排序保证可复现性
    classes = sorted(
        [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    )
    for class_name in classes:
        class_dir = os.path.join(data_dir, class_name)
        for file_name in os.listdir(class_dir):
            # 只保留常见图片格式
            if file_name.lower().endswith((".jpg", ".jpeg", ".png")):
                image_paths.append(os.path.join(class_dir, file_name))
                label_names.append(class_name)
    print(f"从 {data_dir} 加载了 {len(image_paths)} 张图片")
    return image_paths, label_names


def extract_sift_vlad(img, kmeans_model):
    """
    提取 SIFT 关键点并计算 VLAD（局部聚合描述子）向量。

    VLAD 通过累加每个描述子与其最近聚类中心的残差，形成紧凑的图像表示。

    参数
    ----------
    img : numpy.ndarray
        BGR 格式的图像。
    kmeans_model : sklearn.cluster.KMeans
        已训练的码本（聚类中心）。

    返回
    -------
    numpy.ndarray
        VLAD 特征向量，维度 = 聚类数 × 128。
        若无关键点，返回零向量。
    """
    sift = cv2.SIFT_create()
    kp, des = sift.detectAndCompute(img, None)
    # 处理无特征点的情况，返回同维度的零向量以保证特征维度一致
    if des is None or len(des) == 0:
        return np.zeros(kmeans_model.cluster_centers_.shape[0] * 128, dtype=np.float32)

    # 将每个描述子分配到最近的码本中心（硬分配）
    labels, _ = vq(des, kmeans_model.cluster_centers_)
    k = kmeans_model.n_clusters  # 通常取 64
    d = des.shape[1]  # SIFT维度固定为 128
    vlad = np.zeros((k, d), dtype=np.float32)

    # 累加残差：描述子向量减去其对应聚类中心
    for i, (label, desc) in enumerate(zip(labels, des)):
        vlad[label] += desc - kmeans_model.cluster_centers_[label]

    # 展平后做 L2 归一化，使特征模长一致，降低光照变化的影响
    vlad = vlad.flatten()
    vlad = vlad / (np.linalg.norm(vlad) + 1e-8)
    return vlad


def extract_spm_hog(img):
    """
    提取 HOG 特征，输入图像先经 Letterbox 等比例缩放+填充。

    目的：保持物体形状（不拉伸），同时固定输入尺寸为 256x256。
    填充区域为黑色，其梯度为零，不影响 HOG 统计。

    参数
    ----------
    img : numpy.ndarray
        BGR 格式的图像。

    返回
    -------
    numpy.ndarray
        HOG 特征向量。
    """
    # ---------- Letterbox 处理：防止直接拉伸导致物体变形 ----------
    target_size = 256
    h, w = img.shape[:2]
    scale = target_size / max(h, w)  # 长边缩放到目标尺寸
    new_w = int(w * scale)
    new_h = int(h * scale)
    new_w = max(1, new_w)
    new_h = max(1, new_h)

    # 等比例缩放，使用 INTER_AREA 适合缩小，能减少锯齿
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    # 创建黑色画布（填充值为0），因为黑色梯度为零，不会干扰HOG统计
    canvas = np.full((target_size, target_size, 3), 0, dtype=np.uint8)
    # 居中放置缩放后的图像
    dx = (target_size - new_w) // 2
    dy = (target_size - new_h) // 2
    canvas[dy : dy + new_h, dx : dx + new_w] = resized

    # ---------- 提取 HOG ----------
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
    提取两个形状描述子：长宽比和凸包面积比。

    - 长宽比（高度/宽度）：反映物体垂直或水平走向，区分塔（高）与建筑（宽）。
    - 凸包面积比（物体面积 / 凸包面积）：反映形状的紧凑度，规整建筑该值接近1。

    参数
    ----------
    img : numpy.ndarray
        BGR 格式的图像。

    返回
    -------
    numpy.ndarray
        长度为2的数组 [aspect, area_ratio]。
    """
    h, w = img.shape[:2]
    aspect = h / (w + 1e-6)  # 避免除以零

    # 二值化获取前景区域（阈值30可过滤掉极暗噪声）
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        # 无轮廓时，面积比设为1（表示无凸包或主体不可见）
        return np.array([aspect, 1.0], dtype=np.float32)

    # 取最大轮廓作为主体（假设地标占画面主要部分）
    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    area_ratio = area / (hull_area + 1e-6)
    return np.array([aspect, area_ratio], dtype=np.float32)


def extract_color_moments(img):
    """
    在 Lab 颜色空间计算一阶（均值）、二阶（标准差）、三阶（偏度）矩。

    每个通道贡献3个值，共 3通道 × 3矩 = 9 维特征。
    颜色矩对光照变化有一定鲁棒性，适合描述自然景观的整体色调。

    参数
    ----------
    img : numpy.ndarray
        BGR 格式的图像。

    返回
    -------
    numpy.ndarray
        长度为9的颜色矩特征向量。
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)  # Lab更接近人类视觉感知
    moments = []
    for ch in range(3):
        channel = lab[:, :, ch].astype(np.float32)
        mean = np.mean(channel)
        std = np.std(channel)
        # 偏度：三阶中心矩除以标准差的三次方，描述分布对称性
        skew = np.mean(((channel - mean) ** 3)) / ((std + 1e-8) ** 3)
        moments.extend([mean, std, skew])
    return np.array(moments, dtype=np.float32)


def extract_glcm(img):
    """
    计算灰度共生矩阵（GLCM）的四个纹理统计量。

    统计量包括：
        - energy（能量）       ：纹理均匀性（灰度分布越均匀，能量越高）
        - contrast（对比度）   ：局部灰度变化程度（边缘越强，对比度越高）
        - homogeneity（同质性）：图像平滑度（越平滑，同质性越高）
        - correlation（相关性）：灰度线性依赖关系

    为加速计算，先将灰度图缩放至 128x128（经验值，平衡效率与稳定性）。
    使用两个方向（0°和45°）取平均，获得一定的旋转不变性。

    参数
    ----------
    img : numpy.ndarray
        BGR 格式的图像。

    返回
    -------
    numpy.ndarray
        长度为4的 GLCM 特征向量。
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 缩放至固定尺寸，避免原图过大导致计算缓慢；使用 INTER_AREA 保留纹理分布
    gray = cv2.resize(gray, (128, 128), interpolation=cv2.INTER_AREA)

    # 计算 GLCM：距离=1，角度为0°和45°，对称矩阵，灰度级256
    glcm = graycomatrix(
        gray, distances=[1], angles=[0, np.pi / 4], levels=256, symmetric=True
    )
    props = ["energy", "contrast", "homogeneity", "correlation"]
    features = []
    for prop in props:
        # 对两个角度取平均，减小方向敏感性
        feat = graycoprops(glcm, prop).mean()
        features.append(feat)
    return np.array(features, dtype=np.float32)


def extract_features(img_path, kmeans_model):
    """
    从单张图片路径提取所有特征，并将其拼接成一个完整的一维特征向量。

    特征组成顺序：[VLAD, HOG, 几何, 颜色矩, GLCM]。
    此函数是特征提取的总入口，依次调用各个子特征提取函数。

    参数
    ----------
    img_path : str
        图片文件路径。
    kmeans_model : sklearn.cluster.KMeans
        已训练的 VLAD 码本。

    返回
    -------
    numpy.ndarray
        拼接后的完整特征向量。
    """
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"无法读取图片: {img_path}")

    # 支路A：结构特征（适合建筑、塔等刚性地标）
    vlad = extract_sift_vlad(img, kmeans_model)  # 局部特征聚合
    hog = extract_spm_hog(img)  # 全局边缘方向
    geom = extract_geometry(img)  # 形状先验

    # 支路B：颜色与纹理特征（适合自然景观）
    color = extract_color_moments(img)  # 颜色分布统计
    glcm = extract_glcm(img)  # 纹理粗糙度

    # 将所有特征拼接为最终特征向量
    feat = np.concatenate([vlad, hog, geom, color, glcm])
    return feat.astype(np.float32)


if __name__ == "__main__":
    # ---------- 模型保存目录 ----------
    TRAIN_DIR = "./data/train"
    VALID_DIR = "./data/valid"
    MODEL_DIR = "./models"
    os.makedirs(MODEL_DIR, exist_ok=True)

    # ---------- 加载数据集 ----------
    print("=" * 50)
    train_paths, train_names = load_data(TRAIN_DIR)
    val_paths, val_names = load_data(VALID_DIR)

    # ---------- 标签编码 ----------
    # 只使用训练集拟合编码器，验证集标签必须属于训练集类别
    encoder = LabelEncoder()
    train_labels = encoder.fit_transform(train_names)
    try:
        val_labels = encoder.transform(val_names)
    except ValueError as e:
        raise ValueError("验证集中存在训练集未包含的类别，请检查数据") from e
    # 保存编码器，供后续 test.py 解码预测结果
    with open(os.path.join(MODEL_DIR, "label_encoder.pkl"), "wb") as f:
        pickle.dump(encoder, f)
    print(f"训练集类别数: {len(encoder.classes_)}")

    # ---------- 训练 VLAD 码本 ----------
    # 先收集所有训练图片的 SIFT 描述子，用于 KMeans 聚类
    print("\n" + "=" * 50)
    print("正在使用流式 MiniBatchKMeans 训练 VLAD 码本...")

    # 初始化聚类器
    kmeans = MiniBatchKMeans(
        n_clusters=64,
        batch_size=10000,  # 每次迭代使用的内部子集大小
        random_state=42,
        n_init=3,
        max_iter=100,
    )
    sift = cv2.SIFT_create()

    processed_images = 0  # 实际提取到特征并参与训练的图片数
    total_des_used = 0  # 累计参与训练的描述子总数（用于观察）
    max_des_per_image = 300  # 限制单张图片最多抽取关键点，加速训练并平衡样本

    for idx, path in enumerate(train_paths):
        img = cv2.imread(path)
        if img is None:
            continue
        _, des = sift.detectAndCompute(img, None)
        if des is not None and len(des) > 0:
            # 对单张图片的描述子进行随机下采样，避免某张图特征过多影响码本均衡
            if len(des) > max_des_per_image:
                indices = np.random.choice(len(des), max_des_per_image, replace=False)
                des = des[indices]
            # 核心：增量更新，不存储任何历史描述子
            kmeans.partial_fit(des)
            processed_images += 1
            total_des_used += len(des)

        if (idx + 1) % 100 == 0:
            print(
                f"已处理 {idx + 1} 张图片，其中 {processed_images} 张参与训练，累计增量训练 {total_des_used} 个描述子"
            )

    print(
        f"总计处理 {idx + 1} 张图片，其中 {processed_images} 张参与训练，累计使用 {total_des_used} 个描述子完成码本训练"
    )

    # 保存训练好的码本
    with open(os.path.join(MODEL_DIR, "kmeans_vlad.pkl"), "wb") as f:
        pickle.dump(kmeans, f)
    print("VLAD 码本训练完成")

    # ---------- 提取训练集特征 ----------
    print("\n" + "=" * 50)
    print("开始提取训练集特征...")
    X_train = []
    for path in train_paths:
        feat = extract_features(path, kmeans)
        X_train.append(feat)
    X_train = np.array(X_train)
    y_train = np.array(train_labels)

    # ---------- 特征标准化 ----------
    # 标准化使各维度均值为0，方差为1，消除不同特征量纲差异的影响
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    with open(os.path.join(MODEL_DIR, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)

    # ---------- 训练 SVM ----------
    print("训练 SVM 分类器...")
    # 使用 RBF 核处理非线性问题，开启概率估计以便后续贝叶斯决策
    svm = SVC(kernel="rbf", C=1.0, gamma="scale", random_state=42)
    svm.fit(X_train_scaled, y_train)
    with open(os.path.join(MODEL_DIR, "svm_model.pkl"), "wb") as f:
        pickle.dump(svm, f)

    # ---------- 验证集评估 ----------
    print("开始提取验证集特征...")
    X_val = []
    for path in val_paths:
        feat = extract_features(path, kmeans)
        X_val.append(feat)
    X_val = np.array(X_val)
    X_val_scaled = scaler.transform(X_val)
    y_val = np.array(val_labels)
    y_pred = svm.predict(X_val_scaled)

    # 输出整体准确率和每个类别的详细指标
    acc = accuracy_score(y_val, y_pred)
    print(f"验证集准确率: {acc:.4f}")
    report = classification_report(y_val, y_pred, target_names=encoder.classes_)
    print(report)

    print(f"训练完成，所有模型已保存至 {MODEL_DIR}")
