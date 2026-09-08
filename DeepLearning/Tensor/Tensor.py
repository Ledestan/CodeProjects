import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
from torchvision import transforms


def load_image_pil(path):
    """使用 PIL 读取图像并显示"""
    try:
        print("PIL 读取图像")
        image = Image.open(path).convert("RGB")
        plt.imshow(image)
        plt.axis("off")
        plt.title("Image")
        plt.show()
        print(f"图片形状: {image.size}")  # [W, H]
        print(f"图片类型: {type(image)}")
        return image
    except Exception as e:
        print(f"无法加载图片: {e}")
        return None


def load_image_cv2(path):
    """使用 OpenCV 读取图像并显示"""
    try:
        print("OpenCV 读取图像")
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        cv2.imshow("Image", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        print(f"图片形状: {image.shape}")  # [H, W, C]
        print(f"图片类型: {type(image)}")
        return image
    except Exception as e:
        print(f"无法加载图片: {e}")
        return None


def load_image(path):
    """使用 PIL 读取图像"""
    try:
        image = Image.open(path).convert("RGB")
        return image
    except Exception as e:
        print(f"无法加载图片: {e}")
        return None


def image_to_tensor(image):
    """PIL 图像转换为 PyTorch 张量"""
    transform = transforms.ToTensor()
    tensor = transform(image)
    print(f"张量形状: {tensor.shape}")  # [C, H, W]
    return tensor


def tensor_to_image(tensor):
    """PyTorch 张量转换为 PIL 图像"""
    tensor = torch.clamp(tensor, 0, 1)  # 确保范围在 [0, 1]
    image_np = tensor.permute(1, 2, 0).numpy()  # [C, H, W] -> (H, W, C)
    image_np = (image_np * 255).astype(np.uint8)  # 0~1 浮点数 -> 0~255 uint8
    image = Image.fromarray(image_np)  # NumPy 数组转 PIL
    return image


def show_image(image, title="Image"):
    """使用 matplotlib 显示图像"""
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(image)
    ax.set_title(title, fontsize=12)
    ax.axis("off")
    plt.show()


if __name__ == "__main__":
    path = "data/image.png"

    # 演示 PIL 读取并显示
    pil_img = load_image_pil(path)

    # 演示 OpenCV 读取并显示
    cv2_img = load_image_cv2(path)

    # 使用 PIL 读取图像
    original_img = load_image(path)

    # 转换为张量
    tensor_img = image_to_tensor(original_img)

    # 张量再转回图像
    restored_img = tensor_to_image(tensor_img)

    # 显示恢复后的图像
    show_image(restored_img, title="Restored Image")