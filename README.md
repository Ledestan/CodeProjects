

# CodeProject

这是一个包含多个子项目的综合代码仓库，涵盖机器学习、图像处理、数据分析和游戏开发等领域。

## 项目列表

### 1. ChessGame（中国象棋游戏）
- `ChessGame.py` - 游戏主程序
- `ChessBoard.py` - 棋盘类，管理棋盘绘制和操作
- `ChessPiece.py` - 棋子类及各种棋子的移动策略

实现了完整的中国象棋游戏，包含将、士、相、车、马、炮、兵等棋子的走法规则。

### 2. ClusterAnalysis（聚类分析）
- `ClusterAnalysis_788.py` - 788点聚类分析
- `ClusterAnalysis_80.py` - 80点聚类分析  
- `CustomerAnalysis.py` - 客户聚类分析

支持 K-Means、GMM、DBSCAN 等聚类算法，包含肘部法则和轮廓系数评估。

### 3. DataAnalysis（数据分析）
- `DataAnalysis.py` - 薪资数据分析

包含数据预览、重复值处理、缺失值处理、特征提取和互信息分析等功能。

### 4. DimReduction（降维分析）
- `PCA_Iris.py` - Iris 数据集 PCA 降维
- `PCA_Face.py` - 人脸图像 PCA 降维
- `FaceRecAnalysis.py` - 人脸识别综合分析

实现了 PCA、LDA 等降维算法，用于人脸识别和分析。

### 5. ImageClassifier（图像分类器）
- `ImageClassifier.py` - 基于 LBP 特征和神经网络的图像分类器
- `train.py` - 训练脚本
- `test.py` - 测试脚本

使用 LBP 特征提取和神经网络进行猫狗图像分类。

### 6. ImageProcessing（图像处理）
- `ImageProcessing.py` - 图像处理工具

支持各种图像变换和 SVD 压缩。

### 7. ImageRecognition（图像识别）
- `ImageRecognition.py` - 图像识别工具

### 8. IndustrialHeritage（工业遗产保护系统）
- Flask Web 应用
- `app.py` - 主程序
- `image_recognizer.py` - 图像识别模块
- `qa_engine.py` - 问答引擎

包含工业遗产知识问答和图像识别功能的 Web 系统。

### 9. RollCall（点名系统）
- `RollCall.py` - 数字点名系统

### 10. SurveryAnalysis（问卷分析）
- `SurveryAnalysis.py` - 问卷调查分析

## 环境要求

- Python 3.x
- numpy
- pandas
- matplotlib
- scikit-learn
- flask
- pygame
- opencv-python

## 安装

```bash
pip install numpy pandas matplotlib scikit-learn flask pygame opencv-python
```

## 各项目使用说明

### ChessGame
```bash
cd ChessGame
python ChessGame.py
```

### ClusterAnalysis
```python
from ClusterAnalysis.CustomerAnalysis import CustomerAnalysis
ca = CustomerAnalysis('data/Mall_Customers.csv')
ca.show_clusters()
```

### ImageClassifier
```bash
# 训练
cd ImageClassifier
python train.py

# 测试
python test.py
```

### IndustrialHeritage
```bash
cd IndustrialHeritage
python app.py
```

## 许可证

MIT License

## 作者

LTingchu