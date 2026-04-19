import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

class ImageProcessing:
    def __init__(self, path:str):
        """读取图像数据转换为 NumPy 数组"""
        self.image_orig = plt.imread(path)
        self.image_gray = None

    def transform(self):
        """彩色图转换灰度图"""
        # float32 转 uint8
        self.image_orig = (self.image_orig * 255).astype(np.uint8)

        # 提取 RGB 通道
        R = self.image_orig[:, :, 0]
        G = self.image_orig[:, :, 1]
        B = self.image_orig[:, :, 2]

        # 加权平均并转为 uint8
        image_gray = 0.299 * R + 0.587 * G + 0.114 * B
        self.image_gray = np.clip(image_gray, 0, 255).astype(np.uint8)

    def show(self):
        """图像展示"""

    def preview(self):
        """预览图像矩阵信息"""
        # 彩色图像
        print('彩色图，左上角 2 * 3 像素信息：')
        print(self.image_orig[:2, :3])
        print(f'矩阵信息：{self.image_orig.shape}')
        print(f'数据类型：{self.image_orig.dtype}')

        # 灰度图像
        print('灰度图，左上角 5 * 5 像素信息：')
        print(self.image_gray[:5, :5])
        print(f'矩阵信息：{self.image_gray.shape}')
        print(f'数据类型：{self.image_gray.dtype}')

        # 显示图像
        plt.figure(figsize=(10, 4))  # 设置画布大小

        # 显示彩色图
        plt.subplot(1, 2, 1)
        plt.imshow(self.image_orig)
        plt.title('Original Image')
        plt.axis('off')
        
        # 显示灰度图
        plt.subplot(1, 2, 2)
        plt.imshow(self.image_gray, cmap='gray')
        plt.title('Grayscale Image')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()

    def symmetry(self):
        """对称变换"""

    def zoom(self):
        """缩放变换"""

    def cut(self):
        """剪切变换"""

    def rotate(self):
        """旋转变换"""

    def translation(self):
        """平移变换"""

    def composite(self):
        """复合变换"""

if __name__ == "__main__":
    image_path = 'static/image.png'
    imgproc = ImageProcessing(image_path)
    imgproc.transform()
    imgproc.preview()
    