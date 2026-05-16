# CodeProject

This is a comprehensive code repository containing multiple subprojects covering areas such as machine learning, image processing, data analysis, and game development.

## Project List

### 1. ChessGame (Chinese Chess Game)
- `ChessGame.py` - Main game program
- `ChessBoard.py` - Chessboard class managing board rendering and operations
- `ChessPiece.py` - Chess piece class and movement strategies for all piece types

Implements a complete Chinese chess game with rules for all pieces: General, Advisor, Elephant, Rook, Horse, Cannon, and Pawn.

### 2. ClusterAnalysis
- `ClusterAnalysis_788.py` - 788-point clustering analysis
- `ClusterAnalysis_80.py` - 80-point clustering analysis
- `CustomerAnalysis.py` - Customer clustering analysis

Supports clustering algorithms including K-Means, GMM, and DBSCAN, with evaluation via the elbow method and silhouette coefficient.

### 3. DataAnalysis
- `DataAnalysis.py` - Salary data analysis

Includes functions for data preview, duplicate value handling, missing value imputation, feature extraction, and mutual information analysis.

### 4. DimReduction
- `PCA_Iris.py` - PCA dimensionality reduction on the Iris dataset
- `PCA_Face.py` - PCA dimensionality reduction on face images
- `FaceRecAnalysis.py` - Comprehensive face recognition analysis

Implements dimensionality reduction algorithms such as PCA and LDA for face recognition and analysis.

### 5. ImageClassifier
- `ImageClassifier.py` - Image classifier based on LBP features and neural networks
- `train.py` - Training script
- `test.py` - Testing script

Performs cat and dog image classification using LBP feature extraction and neural networks.

### 6. ImageProcessing
- `ImageProcessing.py` - Image processing utilities

Supports various image transformations and SVD-based compression.

### 7. ImageRecognition
- `ImageRecognition.py` - Image recognition tool

### 8. IndustrialHeritage
- Flask web application
- `app.py` - Main application
- `image_recognizer.py` - Image recognition module
- `qa_engine.py` - Question-answering engine

A web system featuring industrial heritage knowledge Q&A and image recognition capabilities.

### 9. RollCall
- `RollCall.py` - Digital attendance system

### 10. SurveryAnalysis
- `SurveryAnalysis.py` - Survey analysis tool

## Requirements

- Python 3.x
- numpy
- pandas
- matplotlib
- scikit-learn
- flask
- pygame
- opencv-python

## Installation

```bash
pip install numpy pandas matplotlib scikit-learn flask pygame opencv-python
```

## Usage Instructions

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
# Train
cd ImageClassifier
python train.py

# Test
python test.py
```

### IndustrialHeritage
```bash
cd IndustrialHeritage
python app.py
```

## License

MIT License

## Author

LTingchu