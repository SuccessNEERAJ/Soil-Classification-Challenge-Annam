#!/bin/bash

# This script downloads the soil classification dataset from Kaggle
# Author: Neeraj Jaiswal
# Team: The Revengers

# Check if kaggle command is available
if ! command -v kaggle &> /dev/null; then
    echo "Error: Kaggle CLI not found. Please install it using 'pip install kaggle'"
    echo "Then configure your Kaggle API credentials in ~/.kaggle/kaggle.json"
    exit 1
fi

# Dataset slug for the soil classification challenge
KAGGLE_DATASET="annam-ai/soilclassification"
TARGET_DIR="./data"
DATASET_DIR="$TARGET_DIR/soil-classification"

echo "===== Downloading soil classification dataset ====="
echo "Dataset: $KAGGLE_DATASET"
mkdir -p "$DATASET_DIR"

# Download the dataset from Kaggle
echo "Downloading dataset..."
kaggle datasets download -d "$KAGGLE_DATASET" -p "$TARGET_DIR" --unzip

# Organize the dataset structure if needed
echo "Organizing dataset files..."
if [ -d "$TARGET_DIR/train" ]; then
    echo "Moving training data to proper location..."s
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

echo "===== Download and organization complete ====="
echo "Dataset files saved to: $DATASET_DIR"
echo "The dataset is now ready for model training and inference."
echo "Run the training notebook to train the soil classification model."
echo "Author: Neeraj Jaiswal (Team: The Revengers)"
