"""
图像识别工具

创建日期: 2026-03-06
需求文件: data

依赖库:
opencv-python>=4.12.0.88
matplotlib>=3.10.8
"""

import os
import cv2
import matplotlib.pyplot as plt

FACE_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False


def detection(image, target_size):
    # 加载检测器
    face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
    # 转灰度图 
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 检测人脸
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    # 无人脸返回 None
    if len(faces) == 0:
        return None
    # 取面积最大的人脸
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    # 裁剪区域
    face_roi = gray[y:y+h, x:x+w]
    # 缩放到目标尺寸
    face_resized = cv2.resize(face_roi, target_size)
    # 归一化像素值
    face_normalized = cv2.normalize(face_resized, None, 0, 255, cv2.NORM_MINMAX)
    
    return face_normalized


class ImageRecognition:
    def __init__(self, path:str):
        self.path = path
        self.image_paths = []
        self.images = []
        self.faces = []
        self.gray_images = []

    def read(self):
        """从目标路径读取图片"""
        for img_name in os.listdir(self.path):
            if img_name.endswith((".jpg", ".png", ".jpeg")):
                self.image_paths.append(os.path.join(self.path, img_name))

    def transform(self):
        """转换图片并检测人脸"""
        for img_path in self.image_paths:
            img = cv2.imread(img_path)
            if img is None:
                print(f"无效图片: {img_path}")
                continue
            img = cv2.resize(img, (640, 480))
            self.images.append(img)
            face = detection(img, (100, 100))
            if face is not None:
                self.faces.append(face)

    def show(self):
        """展示图片"""
        plt.figure(figsize=(10, 5))
        for i, img in enumerate(self.images):
            plt.subplot(3, 4, i + 1)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            plt.imshow(img_rgb)
            plt.axis("off")
            plt.title(os.path.basename(self.image_paths[i]))
        plt.tight_layout()
        plt.show()

    def preview(self):
        """预览灰度图、直方图"""
        plt.figure(figsize=(16, 8))
        for i, img in enumerate(self.images):
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            hist = cv2.calcHist([gray_img], [0], None, [256], [0, 256])
            
            # 灰度图展示
            plt.subplot(len(self.images), 2, i * 2 + 1)
            plt.imshow(gray_img, cmap="gray")
            plt.axis("off")
            plt.title(f"{os.path.basename(self.image_paths[i])} - 灰度图")

            # 直方图展示
            plt.subplot(len(self.images), 2, i * 2 + 2)
            plt.plot(hist, color="gray")
            plt.xlim([0, 256])
            if i == 0:
                plt.ylabel("像素数")
            plt.xlabel("灰度级")
            plt.title(f"{os.path.basename(self.image_paths[i])} - 直方图")

        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(16, 8))
        for i, img in enumerate(self.images):
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            hist = cv2.calcHist([gray_img], [0], None, [256], [0, 256])

            # 直方图均衡化
            equalized_img = cv2.equalizeHist(gray_img)
            equalized_hist = cv2.calcHist([equalized_img], [0], None, [256], [0, 256])
            
            # 均衡化灰度图展示
            plt.subplot(len(self.images), 2, i * 2 + 1)
            plt.imshow(equalized_img, cmap="gray")
            plt.axis("off")
            plt.title(f"{os.path.basename(self.image_paths[i])} - 灰度图")

            # 均衡化直方图展示
            plt.subplot(len(self.images), 2, i * 2 + 2)
            plt.plot(equalized_hist, color="gray")
            plt.xlim([0, 256])
            if i == 0:
                plt.ylabel("像素数")
            plt.xlabel("灰度级")
            plt.title(f"{os.path.basename(self.image_paths[i])} - 直方图")

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    recog = ImageRecognition("data")
    recog.read()
    recog.transform()
    recog.show()
    recog.preview()