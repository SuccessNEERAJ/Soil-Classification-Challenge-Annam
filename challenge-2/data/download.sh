#!/bin/bash

# This script downloads the soil classification part 2 dataset from Kaggle
# Author: Neeraj Jaiswal
# Team: The Revengers

# Check if kaggle command is available
if ! command -v kaggle &> /dev/null; then
    echo "Error: Kaggle CLI not found. Please install it using 'pip install kaggle'"
    echo "Then configure your Kaggle API credentials in ~/.kaggle/kaggle.json"
    exit 1
fi

# Dataset slug for the soil classification challenge part 2
KAGGLE_DATASET="annam-ai/soilclassification-part2"
TARGET_DIR="./data"
DATASET_DIR="$TARGET_DIR/soil-classification-part-2"

echo "===== Downloading soil classification part 2 dataset ====="
echo "Dataset: $KAGGLE_DATASET"
mkdir -p "$DATASET_DIR"

# Download the dataset from Kaggle
echo "Downloading dataset..."
kaggle datasets download -d "$KAGGLE_DATASET" -p "$TARGET_DIR" --unzip

# Organize the dataset structure if needed
echo "Organizing dataset files..."
if [ -d "$TARGET_DIR/train" ]; then
    echo "Moving training data to proper location..."
    mkdir -p "$DATASET_DIR/train"
    mv "$TARGET_DIR/train"/* "$DATASET_DIR/train/" 2>/dev/null
fi

if [ -d "$TARGET_DIR/test" ]; then
    echo "Moving test data to proper location..."
    mkdir -p "$DATASET_DIR/test"
    mv "$TARGET_DIR/test"/* "$DATASET_DIR/test/" 2>/dev/null
fi

# Move any CSV files to the dataset directory
echo "Moving metadata files..."
mv "$TARGET_DIR"/*.csv "$DATASET_DIR/" 2>/dev/null

# Create model directories
echo "Creating model directories..."
mkdir -p "$DATASET_DIR/models"

echo "===== Download and organization complete ====="
echo "Dataset files saved to: $DATASET_DIR"
echo "The dataset is now ready for model training and inference."
echo "Note: This solution uses ResNet50 for feature extraction and One-Class SVM for classification."
echo "After running the training notebook, model files will be saved to: $DATASET_DIR/models"
echo "Author: Neeraj Jaiswal (Team: The Revengers)"
