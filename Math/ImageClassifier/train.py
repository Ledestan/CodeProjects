"""
分类器训练脚本
"""

import os
import sys
import numpy as np

sys.dont_write_bytecode = True
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from ImageClassifier import ImageClassifier


def main():
    # 初始化分类器
    clf = ImageClassifier()

    # 缓存文件路径
    cache_file = "data/models/train_features_lbp.npz"

    # 尝试从缓存加载特征
    X_train, y_train = clf.load_features(cache_file)

    if X_train is None:
        # 缓存不存在，需要加载图片并提取特征
        train_dir = "D:/Tingchu/Downloads/kagglecatsanddogs_5340/PetImages"
        # train_dir = "data/train"
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
        verbose=True
    )

    # 保存最优参数
    param_path = "data/models/params.npz"
    clf.save_params(param_path)

    # 输出总结
    print("\n" + "=" * 50)
    print(f"训练样本数: {len(y_train)}")
    print(f"最终损失: {final_loss:.6f}")
    print(f"参数已保存至: {param_path}")


if __name__ == '__main__':
    main()