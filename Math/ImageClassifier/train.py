"""
train.py 分类器训练脚本
运行此脚本将使用 data/train 中的图片训练模型, 并将参数保存到 models/params.npz
"""

import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from ImageClassifier import ImageClassifier


def main():
    # 初始化分类器
    clf = ImageClassifier()

    # 加载训练数据
    train_dir = os.path.join('data', 'train')
    if not os.path.isdir(train_dir):
        print(f"错误：训练目录 '{train_dir}' 不存在，请按 data/train/cat 和 data/train/dog 的结构放置图片。")
        return

    print("正在加载训练图片...")
    train_imgs, train_labels = clf.load_images(train_dir)

    if len(train_imgs) == 0:
        print("错误：训练集中未找到任何图片，请检查 data/train/cat 和 data/train/dog 目录。")
        return

    # 提取特征
    print("正在提取特征...")
    X_train = clf.prepare_dataset(train_imgs)
    y_train = np.array(train_labels)

    # 训练模型
    print("开始训练...")
    w_opt, b_opt, final_loss = clf.train(
        X_train,
        y_train,
        learning_rate=0.01,
        tolerance=1e-5,
        max_iters=5000,
        verbose=True
    )

    # 保存最优参数
    models_dir = 'Models'
    os.makedirs(models_dir, exist_ok=True)
    param_path = os.path.join(models_dir, 'params.npz')
    clf.save_params(param_path)

    # 输出简要总结
    print("\n" + "=" * 50)
    print("训练总结")
    print("=" * 50)
    print(f"训练样本数: {len(y_train)}")
    print(f"最优参数: w = {w_opt}, b = {b_opt:.6f}")
    print(f"最终损失: {final_loss:.6f}")
    print(f"参数已保存至: {param_path}")
    print("训练完成，可运行 test.py 进行图形化预测。")


if __name__ == '__main__':
    main()