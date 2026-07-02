import os
import pickle
import time
import warnings

import cv2
import numpy as np
from scipy.cluster.vq import vq
from skimage.feature import graycomatrix, graycoprops
from sklearn.cluster import MiniBatchKMeans
from sklearn.exceptions import InconsistentVersionWarning
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore", category=InconsistentVersionWarning)


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
            if file_name.lower().endswith((".jpg", ".jpeg", ".png")):
                image_paths.append(os.path.join(class_dir, file_name))
                label_names.append(class_name)
    print(f"从 {data_dir} 加载了 {len(image_paths)} 张图片")
    return image_paths, label_names


def extract_sift_vlad(img, kmeans_model, max_kp=100):
    """
    提取 SIFT 关键点并计算 VLAD（局部聚合描述子）向量。

    SIFT 是一种尺度不变特征变换算法，检测图像中的关键点并生成 128 维描述子，
    对旋转、尺度、亮度变化具有鲁棒性。
    VLAD 将每个描述子分配到最近聚类中心，累加描述子与中心的残差，形成紧凑的全局图像表示，
    相比词袋模型保留了更多空间分布信息。

    参数
    ----------
    img : numpy.ndarray
        BGR 格式的图像。
    kmeans_model : sklearn.cluster.KMeans
        已训练的码本（聚类中心）。
    max_kp : int, optional
        单张图片最多参与 VLAD 编码的关键点数量（随机采样），默认 100。

    返回
    -------
    numpy.ndarray
        VLAD 特征向量，维度 = 聚类数 × 128。
        若无关键点，返回零向量。
    """
    sift = cv2.SIFT_create()
    kp, des = sift.detectAndCompute(img, None)
    # 若无特征点，返回同维度零向量，保证特征维度一致
    if des is None or len(des) == 0:
        return np.zeros(kmeans_model.cluster_centers_.shape[0] * 128, dtype=np.float32)

    # 随机采样关键点，控制编码数量，减少计算量
    if len(des) > max_kp:
        indices = np.random.choice(len(des), max_kp, replace=False)
        des = des[indices]

    # 将描述子分配到最近的聚类中心（硬分配）
    labels, _ = vq(des, kmeans_model.cluster_centers_)
    k = kmeans_model.n_clusters
    d = des.shape[1]

    vlad = np.zeros((k, d), dtype=np.float32)

    # 累加描述子与聚类中心的残差
    residuals = des - kmeans_model.cluster_centers_[labels]
    np.add.at(vlad, labels, residuals)

    # 展平并 L2 归一化，降低光照变化影响
    vlad = vlad.flatten()
    vlad = vlad / (np.linalg.norm(vlad) + 1e-8)
    return vlad


def extract_spm_hog(img):
    """
    提取 HOG（方向梯度直方图）特征，输入图像先经 Letterbox 等比例缩放+填充。

    HOG 通过统计图像局部区域的梯度方向直方图来捕捉物体的边缘和形状信息，
    对光照变化和小的空间偏移具有鲁棒性，常用于行人检测等任务。
    Letterbox 处理保持物体原始宽高比，避免拉伸变形，填充黑色区域梯度为零不影响统计。

    参数
    ----------
    img : numpy.ndarray
        BGR 格式的图像。

    返回
    -------
    numpy.ndarray
        HOG 特征向量。
    """
    target_size = 256
    h, w = img.shape[:2]
    scale = target_size / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)
    new_w = max(1, new_w)
    new_h = max(1, new_h)

    # 等比例缩放，使用 INTER_AREA 适合缩小，减少锯齿
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    # 创建黑色画布，居中放置缩放后的图像
    canvas = np.full((target_size, target_size, 3), 0, dtype=np.uint8)
    dx = (target_size - new_w) // 2
    dy = (target_size - new_h) // 2
    canvas[dy : dy + new_h, dx : dx + new_w] = resized

    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    # 计算 HOG 描述子，将图像划分为 cell，在 block 内归一化，增强光照不变性
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

    长宽比（高度/宽度）可以区分高塔和宽体建筑；凸包面积比（轮廓面积 / 凸包面积）
    反映形状的紧凑程度，规整的建筑物该值接近 1，而不规则形状则较小。

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
    aspect = h / (w + 1e-6)

    # 二值化获取前景区域，阈值 30 过滤暗噪声
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return np.array([aspect, 1.0], dtype=np.float32)

    # 取最大轮廓作为主体
    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    area_ratio = area / (hull_area + 1e-6)
    return np.array([aspect, area_ratio], dtype=np.float32)


def extract_color_moments(img):
    """
    在 Lab 颜色空间计算颜色矩：一阶矩（均值）、二阶矩（标准差）、三阶矩（偏度）。

    颜色矩是一种紧凑的颜色特征，不需要量化颜色直方图，每个通道仅三个值，
    对光照和视角变化具有一定鲁棒性。Lab 空间的设计使得颜色差异更符合人眼感知。

    参数
    ----------
    img : numpy.ndarray
        BGR 格式的图像。

    返回
    -------
    numpy.ndarray
        长度为9的颜色矩特征向量（L均值、L标准差、L偏度、a均值、a标准差、a偏度、b均值、b标准差、b偏度）。
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)  # Lab 颜色空间更接近人类视觉感知
    moments = []
    for ch in range(3):
        channel = lab[:, :, ch].astype(np.float32)
        mean = np.mean(channel)
        std = np.std(channel)
        # 偏度描述分布对称性，计算公式：E[(X-μ)^3] / σ^3
        skew = np.mean(((channel - mean) ** 3)) / ((std + 1e-8) ** 3)
        moments.extend([mean, std, skew])
    return np.array(moments, dtype=np.float32)


def extract_glcm(img):
    """
    基于灰度共生矩阵（GLCM）计算四个纹理统计量：能量、对比度、同质性、相关性。

    GLCM 统计图像中一定距离和方向上的像素对出现的概率，从而描述纹理的粗细、方向性等。
    能量反映纹理均匀程度，对比度反映局部变化强度，同质性衡量平滑度，
    相关性描述灰度线性依赖关系。本实现使用 0° 和 45° 两个方向的平均值，以获得一定旋转不变性。

    参数
    ----------
    img : numpy.ndarray
        BGR 格式的图像。

    返回
    -------
    numpy.ndarray
        长度为4的 GLCM 特征向量 [energy, contrast, homogeneity, correlation]。
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 缩放至固定尺寸，使用 INTER_AREA 保留纹理分布，加速计算
    gray = cv2.resize(gray, (128, 128), interpolation=cv2.INTER_AREA)

    # 计算 GLCM：距离1，对称矩阵，灰度级256
    glcm = graycomatrix(
        gray, distances=[1], angles=[0, np.pi / 4], levels=256, symmetric=True
    )
    props = ["energy", "contrast", "homogeneity", "correlation"]
    features = []
    for prop in props:
        # 对两个方向取平均，减小方向敏感性
        feat = graycoprops(glcm, prop).mean()
        features.append(feat)
    return np.array(features, dtype=np.float32)


def extract_features(img_path, kmeans_model, vlad_max_kp=100):
    """
    从单张图片路径提取所有特征并拼接为一个完整特征向量。

    特征组成顺序：[VLAD, HOG, 几何, 颜色矩, GLCM]。
    其中 VLAD 和 HOG 侧重于结构形状，几何提供整体轮廓比例，
    颜色矩和 GLCM 提供颜色与纹理信息，多特征融合提高分类鲁棒性。

    参数
    ----------
    img_path : str
        图片文件路径。
    kmeans_model : sklearn.cluster.KMeans
        已训练的 VLAD 码本。
    vlad_max_kp : int
        传递给 extract_sift_vlad 的 max_kp 参数。
    is_training : bool
        是否为训练模式。若为 True，则对图像进行随机裁剪（70%~90%），
        用于增强模型对局部特征的鲁棒性。

    返回
    -------
    numpy.ndarray
        拼接后的完整特征向量。
    """
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"无法读取图片: {img_path}")

    # 支路A：结构特征（适合建筑、塔等刚性地标）
    vlad = extract_sift_vlad(img, kmeans_model, max_kp=vlad_max_kp)
    hog = extract_spm_hog(img)
    geom = extract_geometry(img)

    # 支路B：颜色与纹理特征（适合自然景观）
    color = extract_color_moments(img)
    glcm = extract_glcm(img)

    feat = np.concatenate([vlad, hog, geom, color, glcm])
    return feat.astype(np.float32)


if __name__ == "__main__":
    TRAIN_DIR = "./data/train"
    VALID_DIR = "./data/valid"
    MODEL_DIR = "./models"
    CACHE_DIR = "./cache"  # 用于缓存提取的特征，避免重复计算
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)

    overall_start = time.time()  # 总运行时间计时开始

    # ========== 数据加载 ==========
    print("========== 数据加载 ==========")
    t_start = time.time()
    train_paths, train_names = load_data(TRAIN_DIR)
    val_paths, val_names = load_data(VALID_DIR)
    print(f"[耗时] 加载数据: {time.time() - t_start:.2f} 秒")

    # ========== 标签编码 ==========
    encoder_path = os.path.join(MODEL_DIR, "label_encoder.pkl")
    if os.path.exists(encoder_path):
        print("\n" + "========== 加载标签编码器 ==========")
        with open(encoder_path, "rb") as f:
            encoder = pickle.load(f)
        print(f"类别数: {len(encoder.classes_)}")
        train_labels = encoder.transform(train_names)
        val_labels = encoder.transform(val_names)
    else:
        print("\n" + "========== 训练标签编码器 ==========")
        t_start = time.time()
        encoder = LabelEncoder()
        train_labels = encoder.fit_transform(train_names)
        try:
            val_labels = encoder.transform(val_names)
        except ValueError as e:
            raise ValueError("验证集中存在训练集未包含的类别，请检查数据") from e
        with open(encoder_path, "wb") as f:
            pickle.dump(encoder, f)
        print(f"训练集类别数: {len(encoder.classes_)}")
        print(f"[耗时] 标签编码: {time.time() - t_start:.2f} 秒")

    # ========== VLAD 码本 ==========
    kmeans_path = os.path.join(MODEL_DIR, "kmeans_vlad.pkl")
    if os.path.exists(kmeans_path):
        print("\n" + "========== 加载 VLAD 码本 ==========")
        with open(kmeans_path, "rb") as f:
            kmeans = pickle.load(f)
        print(f"聚类数: {kmeans.n_clusters}")
    else:
        print("\n" + "========== 训练 VLAD 码本 ==========")
        t_start = time.time()

        kmeans = MiniBatchKMeans(
            n_clusters=64,  # 视觉单词数量，决定 VLAD 向量的维度
            batch_size=10000,
            random_state=42,
            n_init=3,
            max_iter=100,
        )
        sift = cv2.SIFT_create()

        processed_images = 0
        total_des_used = 0
        max_des_per_image = 500  # 限制单张图片抽取的描述子数量，平衡各类别贡献

        for idx, path in enumerate(train_paths):
            img = cv2.imread(path)
            if img is None:
                continue
            _, des = sift.detectAndCompute(img, None)
            if des is not None and len(des) > 0:
                # 随机下采样，避免某张图特征过多影响码本均衡
                if len(des) > max_des_per_image:
                    indices = np.random.choice(
                        len(des), max_des_per_image, replace=False
                    )
                    des = des[indices]
                # 增量更新聚类器，不存储历史描述子
                kmeans.partial_fit(des)
                processed_images += 1
                total_des_used += len(des)

            if (idx + 1) % 50 == 0 or (idx + 1) == len(train_paths):
                print(
                    f"VLAD 码本训练进度: {idx + 1}/{len(train_paths)}，其中 {processed_images} 张参与训练，累计增量训练 {total_des_used} 个描述子"
                )

        print("VLAD 码本训练完成")

        with open(kmeans_path, "wb") as f:
            pickle.dump(kmeans, f)
        t_vlad = time.time() - t_start
        print(f"[耗时] VLAD 码本训练及保存: {t_vlad:.2f} 秒")

    # ========== 特征提取 ==========
    train_cache = os.path.join(CACHE_DIR, "X_train.npy")
    val_cache = os.path.join(CACHE_DIR, "X_val.npy")

    VLAD_MAX_KP = 500  # VLAD 编码时每张图使用的最大关键点数

    # 处理训练集特征
    print("\n" + "========== 训练集特征提取 ==========")
    if os.path.exists(train_cache):
        print("发现训练集缓存特征，直接加载...")
        X_train = np.load(train_cache)
        t_train_feat = time.time() - t_start
        print(f"训练集特征形状: {X_train.shape}")
    else:
        print("未找到训练集缓存，开始提取训练集特征...")
        t_start = time.time()
        X_train = []
        total_train = len(train_paths)
        for idx, path in enumerate(train_paths):
            feat = extract_features(path, kmeans, vlad_max_kp=VLAD_MAX_KP)
            X_train.append(feat)
            # 进度提示：每 50 张或最后一张时输出
            if (idx + 1) % 50 == 0 or (idx + 1) == total_train:
                print(f"训练集特征提取进度: {idx + 1}/{total_train}")
        X_train = np.array(X_train)
        np.save(train_cache, X_train)
        print(f"训练集特征已保存至 {train_cache}，形状: {X_train.shape}")
        print(f"[耗时] 训练集特征提取: {time.time() - t_start:.2f} 秒")

    # 处理验证集特征
    print("\n" + "========== 验证集特征提取 ==========")
    if os.path.exists(val_cache):
        print("发现验证集缓存特征，直接加载...")
        X_val = np.load(val_cache)
        t_val_feat = time.time() - t_start
        print(f"验证集特征形状: {X_val.shape}")
    else:
        print("未找到验证集缓存，开始提取验证集特征...")
        t_start = time.time()
        X_val = []
        total_val = len(val_paths)
        for idx, path in enumerate(val_paths):
            feat = extract_features(path, kmeans, vlad_max_kp=VLAD_MAX_KP)
            X_val.append(feat)
            if (idx + 1) % 50 == 0 or (idx + 1) == total_val:
                print(f"验证集特征提取进度: {idx + 1}/{total_val}")
        X_val = np.array(X_val)
        np.save(val_cache, X_val)
        print(f"验证集特征已保存至 {val_cache}，形状: {X_val.shape}")
        print(f"[耗时] 验证集特征提取: {time.time() - t_start:.2f} 秒")

    y_train = np.array(train_labels)
    y_val = np.array(val_labels)

    # ========== 特征标准化 ==========
    scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
    if os.path.exists(scaler_path):
        print("\n" + "========== 加载标准化器 ==========")
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
        X_train_scaled = scaler.transform(X_train)  # 只用 transform，不重新 fit
        X_val_scaled = scaler.transform(X_val)
        print("标准化器加载完成")
    else:
        print("\n" + "========== 特征标准化 ==========")
        t_start = time.time()
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        with open(scaler_path, "wb") as f:
            pickle.dump(scaler, f)
        print(f"[耗时] 特征标准化: {time.time() - t_start:.2f} 秒")

    # ========== 训练 SVM 分类器 ==========
    svm_path = os.path.join(MODEL_DIR, "svm_model.pkl")
    if os.path.exists(svm_path):
        print("\n" + "========== 加载 SVM 分类器 ==========")
        with open(svm_path, "rb") as f:
            svm = pickle.load(f)
        print("SVM 加载完成")
    else:
        print("\n" + "========== 训练 SVM 分类器 ==========")
        t_start = time.time()
        svm = SVC(kernel="rbf", C=1.0, gamma="scale", random_state=42, verbose=True)
        svm.fit(X_train_scaled, y_train)
        with open(svm_path, "wb") as f:
            pickle.dump(svm, f)
        print(f"[耗时] SVM 训练及保存: {time.time() - t_start:.2f} 秒")

    # ========== 验证集预测 ==========
    print("\n" + "========== 验证集预测 ==========")
    t_start = time.time()
    X_val_scaled = scaler.transform(X_val)
    y_pred = svm.predict(X_val_scaled)
    print(f"[耗时] 验证集预测: {time.time() - t_start:.2f} 秒")

    # ========== 评估 ==========
    print("\n" + "========== 评估模型性能 ==========")
    t_start = time.time()
    acc = accuracy_score(y_val, y_pred)
    print(f"验证集准确率: {acc:.4f}")
    report = classification_report(y_val, y_pred, target_names=encoder.classes_)
    print(report)
    t_eval = time.time() - t_start
    print(f"[耗时] 性能评估: {t_eval:.2f} 秒")

    # ========== 总运行时间 ==========
    overall_time = time.time() - overall_start
    print("\n" + "=" * 50)
    print(f"所有步骤完成，总运行时间: {overall_time:.2f} 秒")
    print(f"模型文件保存在: {MODEL_DIR}")
