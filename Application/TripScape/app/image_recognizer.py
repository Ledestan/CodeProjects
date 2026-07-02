import os
import sys
import tempfile

sys.dont_write_bytecode = True

import cv2
import numpy as np

from .db import LandmarkDB
from .test import LandmarkPredictor


class ImageRecognizer:
    def __init__(self, model_dir="models"):
        self.db = LandmarkDB()
        self.model_dir = model_dir

        # 加载地标信息（用于识别后返回详情）
        self.target_info = {}
        self._load_heritage_info()

        # 初始化新的预测器（内部会自动加载所有模型文件）
        try:
            self.predictor = LandmarkPredictor(model_dir=model_dir)
            print("ImageRecognizer: 新模型加载成功。")
        except Exception as e:
            print(f"ImageRecognizer: 模型加载失败: {e}")
            self.predictor = None

    def _load_heritage_info(self):
        """从数据库加载地标信息，键为小写的 target_id"""
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

    def recognize(self, image_data):
        """
        识别主函数

        :param image_data: 图片二进制数据
        :return: 识别结果字典
        """
        if self.predictor is None:
            return {"success": False, "message": "模型未加载，请检查模型文件是否存在。"}

        try:
            # 解码图片（验证是否有效）
            nparr = np.frombuffer(image_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                return {"success": False, "message": "无法解析图片"}

            # 将图片写入临时文件（predictor 需要路径）
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                tmp_path = tmp.name
                cv2.imwrite(tmp_path, img)

            # 调用新预测器（默认使用贝叶斯滑动窗口，top_k=1 只取最佳结果）
            results = self.predictor.predict(tmp_path, top_k=1, use_bayes=True)

            # 删除临时文件
            try:
                os.unlink(tmp_path)
            except:
                pass

            if not results:
                return {"success": False, "message": "未能识别出地标"}

            pred_label, confidence = results[0]
            target_id = pred_label.lower()  # 数据库键是小写

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
