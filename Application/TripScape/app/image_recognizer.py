"""
图像识别引擎 - 加载训练好的随机森林模型进行预测
"""

import os
import sys

sys.dont_write_bytecode = True

import cv2
import joblib
import numpy as np
from skimage.feature import hog

from .db import LandmarkDB


class ImageRecognizer:
    def __init__(self, model_dir="models"):
        self.db = LandmarkDB()
        self.model_dir = model_dir
        self.model_path = os.path.join(model_dir, "landmark_model.pkl")
        self.scaler_path = os.path.join(model_dir, "scaler.pkl")
        self.encoder_path = os.path.join(model_dir, "label_encoder.pkl")

        self.model = None
        self.scaler = None
        self.label_encoder = None

        # 加载地标信息（用于识别后返回详情）
        self.target_info = {}
        self._load_heritage_info()

        # 加载训练好的模型
        if self._check_files_exist():
            self._load_model()
        else:
            print("警告: 模型文件不存在，请先运行 app/train.py 训练模型。")

    def _load_heritage_info(self):
        """从数据库加载地标信息"""
        rows = self.db.get_heritage_info()
        if rows:
            for row in rows:
                self.target_info[row["target_id"].lower()] = {
                    "name": row["name"],
                    "year": row["year"] or "",
                    "description": row["description"] or "",
                    "location": row["location"] or "",
                    "current_status": row["current_status"] or "",
                }

    def _check_files_exist(self):
        """检查所有必要文件是否存在"""
        return all(
            os.path.exists(p)
            for p in [self.model_path, self.scaler_path, self.encoder_path]
        )

    def _load_model(self):
        """加载模型文件"""
        self.model = joblib.load(self.model_path)
        self.scaler = joblib.load(self.scaler_path)
        self.label_encoder = joblib.load(self.encoder_path)

    def _extract_features(self, image):
        """提取单张图片的特征（与训练时一致）"""
        image = cv2.resize(image, (224, 224))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        hog_feat = hog(
            gray,
            orientations=9,
            pixels_per_cell=(16, 16),
            cells_per_block=(2, 2),
            block_norm="L2-Hys",
            feature_vector=True,
        )

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist(
            [hsv], [0, 1, 2], None, [16, 16, 16], [0, 180, 0, 256, 0, 256]
        )
        hist = cv2.normalize(hist, hist).flatten()

        return np.concatenate([hog_feat, hist])

    def recognize(self, image_data):
        """
        识别主函数
        :param image_data: 图片二进制数据
        :return: 识别结果字典
        """
        if self.model is None:
            return {"success": False, "message": "模型未加载，请先训练模型。"}

        try:
            # 解码图片
            nparr = np.frombuffer(image_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                return {"success": False, "message": "无法解析图片"}

            # 提取特征并标准化
            features = self._extract_features(img).reshape(1, -1)
            features_scaled = self.scaler.transform(features)

            # 预测
            pred_idx = self.model.predict(features_scaled)[0]
            target_id = self.label_encoder.inverse_transform([pred_idx])[0].lower()
            probs = self.model.predict_proba(features_scaled)[0]
            confidence = probs[pred_idx]

            # 查询地标信息
            info = self.target_info.get(target_id)
            if not info:
                return {"success": False, "message": f"未找到地标信息: {target_id}"}

            return {
                "success": True,
                "target_id": target_id,
                "name": info["name"],
                "year": info["year"],
                "description": info["description"],
                "location": info["location"],
                "confidence": float(confidence),
                "confidence_percent": f"{confidence * 100:.1f}%",
                "annotated_image": None,
            }

        except Exception as e:
            return {"success": False, "message": f"识别出错: {str(e)}"}
