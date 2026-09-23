import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
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


def rotate_tensor(tensor, angle):
    """使用 PyTorch transforms 旋转张量（固定角度）并显示对比图"""
    # 使用 RandomRotation 并设置 degrees=(angle, angle)，实现固定角度旋转
    rotater = transforms.RandomRotation(degrees=(angle, angle), expand=True)
    rotated_tensor = rotater(tensor)

    # 将原张量和旋转后的张量都转换为 PIL 图像用于显示
    original_img = tensor_to_image(tensor)
    rotated_img = tensor_to_image(rotated_tensor)

    # 并排显示对比图
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(original_img)
    axes[0].set_title("Original Image", fontsize=12)
    axes[0].axis("off")

    axes[1].imshow(rotated_img)
    axes[1].set_title(f"Rotated by {angle}° (Tensor)", fontsize=12)
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()
    return rotated_tensor


def scale_tensor(tensor, scale_factor):
    """使用 PyTorch transforms 缩放张量（等比）并显示对比图"""
    _, h, w = tensor.shape
    new_h, new_w = int(h * scale_factor), int(w * scale_factor)

    # 使用 Resize 处理张量
    resizer = transforms.Resize((new_h, new_w))
    scaled_tensor = resizer(tensor)

    # 转换用于显示
    original_img = tensor_to_image(tensor)
    scaled_img = tensor_to_image(scaled_tensor)

    # 并排显示对比图
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(original_img)
    axes[0].set_title("Original Image", fontsize=12)
    axes[0].axis("off")

    axes[1].imshow(scaled_img)
    axes[1].set_title(f"Scaled by factor {scale_factor} (Tensor)", fontsize=12)
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()
    return scaled_tensor


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

    # 张量旋转演示
    rotated_tensor = rotate_tensor(tensor_img, 45)

    # 张量缩放演示
    scaled_tensor = scale_tensor(tensor_img, 0.5)
