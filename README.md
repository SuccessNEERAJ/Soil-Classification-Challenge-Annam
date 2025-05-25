# Soil Classification Kaggle Competition Solution

## Team: The Revengers
**Members:** Neeraj Jaiswal, Karanbir Singh, Komal Dadwal, Jatin Mahey, Sonu Choubey  
**Leaderboard Ranks:**
- Challenge 1: Rank 27
- Challenge 2: Rank 16

## Overview
This repository contains our solution to the Soil Classification Kaggle competition. The competition involved analyzing soil images to classify them into different soil types based on visual characteristics like color and texture.

The repository is organized into two main challenges:
1. **Challenge 1:** RGB and texture-based soil classification using Random Forest
2. **Challenge 2:** Anomaly detection in soil images using ResNet50 features and One-Class SVM

## Repository Structure
```
Soil-Classification-Challenge/
├── challenge-1/                 # Standard soil classification solution
│   ├── data/                    # Data directory with download script
│   │   ├── download.sh          # Script to download competition dataset
│   │   ├── soil_attributes_output.csv  # Extracted soil attributes
│   │   └── soil_classifier_pipeline.joblib  # Trained model
│   ├── docs/                    # Documentation
│   ├── notebooks/               # Jupyter notebooks
│   │   ├── training.ipynb       # Model training notebook
│   │   └── inference.ipynb      # Inference notebook
│   ├── src/                     # Source code
│   └── requirements.txt         # Dependencies
│
└── challenge-2/                 # Anomaly detection solution
    ├── data/                    # Data directory with download script
    │   ├── download.sh          # Script to download part-2 dataset
    │   ├── oneclass_svm_model.joblib  # SVM model
    │   ├── pca.joblib           # PCA model for dimensionality reduction
    │   └── scaler.joblib        # Feature scaler
    ├── docs/                    # Documentation
    ├── notebooks/               # Jupyter notebooks
    │   ├── training.ipynb       # Model training notebook
    │   └── inference.ipynb      # Inference notebook
    ├── src/                     # Source code
    └── requirements.txt         # Dependencies
```

## Installation & Setup

### Prerequisites
- Python 3.7+
- Kaggle API credentials (for downloading the dataset)
- Required Python packages

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/soil-classification-challenge.git
cd soil-classification-challenge
```

### 2. Set Up Kaggle API
To download the competition datasets, you need to set up Kaggle API credentials:

1. Create a Kaggle account if you don't have one already
2. Go to your account settings (https://www.kaggle.com/account)
3. Create a new API token and download the `kaggle.json` file
4. Place the `kaggle.json` file in `~/.kaggle/` (Linux/Mac) or `C:\Users\<Windows-username>\.kaggle\` (Windows)
5. Set permissions: `chmod 600 ~/.kaggle/kaggle.json` (Linux/Mac only)

### 3. Install Dependencies
```bash
# For Challenge 1
cd challenge-1
pip install -r requirements.txt

# For Challenge 2
cd ../challenge-2
pip install -r requirements.txt
```

## Running the Solutions

### Challenge 1: Soil Classification

1. **Download the dataset:**
```bash
cd challenge-1/data
bash download.sh
```

2. **Run the notebooks:**
- Open and run `notebooks/training.ipynb` to train the model
- Open and run `notebooks/inference.ipynb` to generate predictions

### Challenge 2: Soil Anomaly Detection

1. **Download the dataset:**
```bash
cd challenge-2/data
bash download.sh
```

2. **Run the notebooks:**
- Open and run `notebooks/training.ipynb` to train the anomaly detection model
- Open and run `notebooks/inference.ipynb` to generate predictions

## Technical Approach

### Challenge 1: Soil Classification
Our approach for the standard soil classification task involved:

1. **Feature extraction:**
   - RGB color features extracted from soil images
   - Texture descriptions based on image processing techniques

2. **Model:**
   - Random Forest Classifier with preprocessing pipeline
   - One-hot encoding for categorical features

3. **Evaluation:**
   - Classification metrics (accuracy, precision, recall, F1-score)
   - Confusion matrix visualization

### Challenge 2: Soil Anomaly Detection
For the anomaly detection task, we implemented:

1. **Feature extraction:**
   - Deep features from pre-trained ResNet50 model
   - Dimensionality reduction using PCA

2. **Model:**
   - One-Class SVM for anomaly detection
   - Standardization of features

3. **Evaluation:**
   - Anomaly detection metrics
   - Visual inspection of anomalies

## Results
Our solutions achieved competitive performance on the Kaggle leaderboards:
- **Challenge 1:** Rank 27 for soil classification using RGB and texture features
- **Challenge 2:** Rank 16 for soil anomaly detection using deep learning features

These results demonstrate our robust approach to both standard classification and anomaly detection in soil images.

## License
This project is available under the MIT License.

## Acknowledgments
- Kaggle for hosting the competition
- The competition organizers for providing the dataset and challenge
- Our team members for their dedication and collaboration
