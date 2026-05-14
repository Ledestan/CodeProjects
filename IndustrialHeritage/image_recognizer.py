import cv2
import numpy as np
import json
import os


class ImageRecognizer:
    def __init__(self):
        self.target_images = {}
        self.target_info = {}
        self.load_targets()
        self.load_heritage_info()
    
    def load_targets(self):
        """加载目标图片（保留原始尺寸）"""
        target_dir = 'data/targets/'
        for filename in os.listdir(target_dir):
            if filename.endswith(('.jpg', '.png', '.jpeg')):
                target_id = filename.split('.')[0]
                path = os.path.join(target_dir, filename)
                
                # 读取图片并转为灰度图
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    # 不再统一缩放，直接存储原始尺寸
                    self.target_images[target_id] = img
    
    def load_heritage_info(self):
        """加载遗产信息"""
        with open('data/heritage_info.json', 'r', encoding='utf-8') as f:
            self.target_info = json.load(f)
    
    def preprocess_image(self, image_data):
        """预处理上传的图片（直方图均衡化，限制最大尺寸）"""
        # 将字节数据转为numpy数组
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            raise ValueError("无法读取图片数据")
        
        # 限制最大尺寸，避免计算过慢（最长边不超过1200像素）
        h, w = img.shape
        max_dim = 1200
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            new_w = int(w * scale)
            new_h = int(h * scale)
            img = cv2.resize(img, (new_w, new_h))
        
        # 直方图均衡化增强对比度
        img_enhanced = cv2.equalizeHist(img)
        
        return img_enhanced
    
    def template_match(self, user_img):
        """多尺度模板匹配算法，返回最佳匹配的模板尺寸"""
        best_match = None
        best_score = 0
        best_location = None
        best_tpl_shape = None  # 记录匹配时模板的实际尺寸 (w, h)
        
        for target_id, target_img in self.target_images.items():
            h_tpl, w_tpl = target_img.shape
            
            # 尝试多种缩放比例，覆盖常见的尺寸变化
            scales = [0.3, 0.5, 0.7, 0.9, 1.0, 1.2, 1.5, 2.0]
            for scale in scales:
                new_w = int(w_tpl * scale)
                new_h = int(h_tpl * scale)
                
                # 缩放后的模板不能大于用户图片
                if new_w > user_img.shape[1] or new_h > user_img.shape[0]:
                    continue
                
                # 缩放模板
                target_resized = cv2.resize(target_img, (new_w, new_h))
                
                # 模板匹配
                result = cv2.matchTemplate(user_img, target_resized, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, max_loc = cv2.minMaxLoc(result)
                
                if max_val > best_score:
                    best_score = max_val
                    best_match = target_id
                    best_location = max_loc
                    best_tpl_shape = (new_w, new_h)
        
        return best_match, best_score, best_location, best_tpl_shape
    
    def recognize(self, image_data):
        """识别主函数"""
        try:
            # 1. 预处理
            user_img = self.preprocess_image(image_data)
            
            # 2. 多尺度模板匹配，获取最佳匹配的模板尺寸
            target_id, score, location, tpl_shape = self.template_match(user_img)
            
            # 3. 判断是否识别成功（阈值可根据实际测试调整）
            if score > 0.50 and target_id in self.target_info:
                info = self.target_info[target_id]
                
                # 4. 在原图上标注（传入匹配到的模板尺寸）
                annotated_img = self.annotate_image(image_data, location, tpl_shape, target_id)
                
                return {
                    "success": True,
                    "target_id": target_id,
                    "name": info["name"],
                    "year": info["year"],
                    "description": info["description"],
                    "location": info["location"],
                    "confidence": float(score),
                    "confidence_percent": f"{score * 100:.1f}%",
                    "annotated_image": annotated_img  # base64编码的标注图
                }
            else:
                return {
                    "success": False,
                    "message": f"未识别到工业遗产（最高相似度：{score:.2%}）",
                    "confidence": float(score)
                }
                
        except Exception as e:
            return {
                "success": False,
                "message": f"识别过程中出错：{str(e)}"
            }
    
    def annotate_image(self, image_data, location, tpl_shape, target_id):
        """在识别成功的图片上添加标注，框大小与实际匹配的模板一致"""
        # 解码原始彩色图片
        nparr = np.frombuffer(image_data, np.uint8)
        color_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # 绘制识别框
        x, y = location
        w, h = tpl_shape
        top_left = (x, y)
        bottom_right = (x + w, y + h)
        
        cv2.rectangle(color_img, top_left, bottom_right, (0, 255, 0), 3)
        
        # 将图片转为base64编码，便于前端显示
        _, buffer = cv2.imencode('.jpg', color_img)
        import base64
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return img_base64


# 全局实例
recognizer = ImageRecognizer()