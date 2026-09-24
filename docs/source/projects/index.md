# Application（应用实践）

## ChessGame（中国象棋游戏）

- **入口文件**：`Application/ChessGame/ChessGame.py`
- **功能描述**：实现了完整的中国象棋游戏，包含将、士、相、车、马、炮、兵等棋子的走法规则。
- **运行方法**：
  ```bash
  cd Application/ChessGame
  python ChessGame.py
  ```

## ImageClassifier（图像分类器）

- **入口文件**：`Application/ImageClassifier/ImageClassifier.py`
- **功能描述**：基于 LBP 特征提取与神经网络，实现猫狗图像分类功能（训练与测试脚本分别为 `train.py` / `test.py`）。
- **运行方法**：
  ```bash
  cd Application/ImageClassifier
  python ImageClassifier.py
  ```

## IndustrialHeritage（工业遗产数字活化）

- **入口文件**：`Application/IndustrialHeritage/app.py`
- **功能描述**：集成图像识别模块与知识问答引擎，用于工业遗产信息查询与保护。
- **运行方法**：
  ```bash
  cd Application/IndustrialHeritage
  python app.py
  ```

## SmartVillage（数治乡音：村级民生诉求分流系统）

- **入口文件**：`Application/SmartVillage/app.py`
- **功能描述**：面向乡村基层治理的轻量级民生诉求闭环管理系统，实现“村民提交、干部分流、大屏公示”全流程数字化。
- **运行方法**：
  ```bash
  cd Application/SmartVillage
  streamlit run app.py
  ```

## SparkNexus（星火云枢：班级智能协同平台）

- **入口文件**：`Application/SparkNexus/run.py`
- **功能描述**：班级管理 Web 应用，支持日程协同、报名签到、通知发布及多角色权限控制（管理员、教师、班委、学生）。
- **运行方法**：
  ```bash
  cd Application/SparkNexus
  python run.py
  ```

## TripScape（行旅识景：世界著名地标智能识别）

- **入口文件**：`Application/TripScape/app/__init__.py`
- **功能描述**：集地标图像识别、AI智能问答、足迹打卡图鉴、动态热力图和成就激励于一体的交互平台。用户上传照片即可识别地标并获取百科信息，输入问题可获得流式AI回答，打卡收集地标卡牌，并在地图上可视化旅行足迹。
- **运行方法**：
  ```bash
  cd Application/TripScape
  python -m app
  ```

---

# DeepLearning（深度学习）

## AD（自动微分）

- **入口文件**：`DeepLearning/AD.py`
- **功能描述**：实现自动微分基础机制，演示前向计算、反向传播与梯度求解流程，用于理解深度学习框架中的自动求导原理。
- **运行方法**：
  ```bash
  cd DeepLearning
  python AD.py
  ```

## DiabetesPrediction（糖尿病预测）

- **入口文件**：`DeepLearning/DiabetesPrediction.py`
- **功能描述**：基于糖尿病相关数据集，完成数据加载、特征处理、模型构建、训练与评估，实现糖尿病预测任务。
- **运行方法**：
  ```bash
  cd DeepLearning
  python DiabetesPrediction.py
  ```

## Tensor（张量）

- **入口文件**：`DeepLearning/Tensor.py`
- **功能描述**：实现张量基础数据结构与常见运算，并结合示例图像数据进行张量读取、转换或操作演示，帮助理解张量在深度学习中的使用方式。
- **运行方法**：
  ```bash
  cd DeepLearning
  python Tensor.py
  ```

---

# PRML（模式识别与机器学习）

## ClassificationModels（分类模型）

- **入口文件**：`PRML/ClassificationModels.py`
- **功能描述**：基于 Kaggle 医疗预约数据集，使用随机森林、逻辑回归预测患者是否按时赴约。
- **运行方法**：
  ```bash
  cd PRML
  python ClassificationModels.py
  ```

## ClusterAnalysis（聚类分析）

- **入口文件**：`PRML/ClusterAnalysis.py`
- **功能描述**：支持 K-Means、GMM、DBSCAN 等聚类算法，包含肘部法则和轮廓系数评估。
- **运行方法**：
  ```bash
  cd PRML
  python ClusterAnalysis.py
  ```

## DataExploration（数据探索）

- **入口文件**：`PRML/DataExploration.py`
- **功能描述**：薪资数据分析，涵盖数据预览、缺失值处理、特征提取及互信息分析。
- **运行方法**：
  ```bash
  cd PRML
  python DataExploration.py
  ```

## DecisionTree（决策树）

- **入口文件**：`PRML/DecisionTree.py`
- **功能描述**：基于西瓜数据集，使用基尼系数（CART）构建决策树，输出树结构与准确率。
- **运行方法**：
  ```bash
  cd PRML
  python DecisionTree.py
  ```

## DimReduction（降维分析）

- **入口文件**：`PRML/DimReduction.py`
- **功能描述**：实现 PCA、LDA 等降维算法，应用于 Iris 数据集及人脸识别场景。
- **运行方法**：
  ```bash
  cd PRML
  python DimReduction.py
  ```

## ImageRecognition（图像识别）

- **入口文件**：`PRML/ImageRecognition.py`
- **功能描述**：基于模板匹配与特征提取的图像识别工具。
- **运行方法**：
  ```bash
  cd PRML
  python ImageRecognition.py
  ```

## LinearRegression（线性回归）

- **入口文件**：`PRML/LinearRegression.py`
- **功能描述**：提供基于梯度下降和 scikit-learn 的线性回归实现，含损失曲线可视化。
- **运行方法**：
  ```bash
  cd PRML
  python LinearRegression.py
  ```

---

# Programming（程序设计）

## 15-Puzzle（数字华容道）

- **入口文件**：`Programming/15-Puzzle.cpp`
- **功能描述**：经典 15 数码（滑动拼图）问题求解，基于 C++ 实现。
- **运行方法**：
  ```bash
  cd Programming
  g++ 15-Puzzle.cpp -o puzzle
  ./puzzle
  ```

## CampusNavigation（校园导航系统）

- **入口文件**：`Programming/CampusNavigation.cpp`
- **功能描述**：基于校园地图数据的路径规划与导航工具。
- **运行方法**：
  ```bash
  cd Programming
  g++ CampusNavigation.cpp -o nav
  ./nav
  ```

## QueueManagement（队列管理系统）

- **入口文件**：`Programming/QueueManagement.cpp`
- **功能描述**：队列数据结构与任务调度管理模拟。
- **运行方法**：
  ```bash
  cd Programming
  g++ QueueManagement.cpp -o queue
  ./queue
  ```

## RollCall（点名系统）

- **入口文件**：`Programming/RollCall.py`
- **功能描述**：数字点名系统，用于随机抽取学生进行课堂签到。
- **运行方法**：
  ```bash
  cd Programming
  python RollCall.py
  ```

## SurveyAnalysis（问卷分析）

- **入口文件**：`Programming/SurveyAnalysis.py`
- **功能描述**：问卷调查数据分析工具。
- **运行方法**：
  ```bash
  cd Programming
  python SurveyAnalysis.py
  ```

## ImageProcessing（图像处理）

- **入口文件**：`Programming/ImageProcessing.py`
- **功能描述**：提供图像变换、SVD 压缩等图像处理工具。
- **运行方法**：
  ```bash
  cd Programming
  python ImageProcessing.py
  ```

## Template（数据结构模板库）

- **入口文件**：`Programming/DoublyLinkedList.py` 与 `Programming/LinkedList.py`
- **功能描述**：手写链表（单链表/双链表）模板，供其他项目复用或参考。
- **运行方法**：无独立运行入口，作为模块导入使用。

---

# 许可证与作者

本项目采用 **MIT 许可证** 开源。

**欢迎参考与学习**
你可以自由地使用、复制、修改和分发本项目的代码。无论是用于个人学习、学术研究还是商业用途，我都表示欢迎。

**使用要求与免责声明**

- **保留声明**：在分发软件副本时，请务必保留原始的版权声明和许可声明。
- **责任限制**：代码按“原样”提供，不提供任何形式的明示或暗示担保。作者不对因使用本软件而产生的任何索赔、损害或其他责任负责。请在使用前自行评估风险。

**作者**：Tingchu
