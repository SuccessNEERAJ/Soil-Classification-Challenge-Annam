"""

Author: Neeraj Jaiswal
Team Name: The Revengers
Team Members: Neeraj Jaiswal, Karanbir Singh, Komal Dadwal, Jatin Mahey, Sonu Choubey
Leaderboard Rank: 16

"""

# Here you add all the preprocessing related details for the task completed from Kaggle.

import pandas as pd
import cv2
import numpy as np
import os
from sklearn.cluster import KMeans
from skimage.feature import local_binary_pattern

def preprocessing():
    # Parameters for texture analysis
    LBP_RADIUS = 3
    LBP_POINTS = 8 * LBP_RADIUS

    # Helper function to get dominant color
    def get_dominant_color(image, k=3):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.reshape((-1, 3))
        kmeans = KMeans(n_clusters=k, random_state=42).fit(image)
        counts = np.bincount(kmeans.labels_)
        dominant_color = kmeans.cluster_centers_[np.argmax(counts)].astype(int)
        return f"rgb({dominant_color[0]}, {dominant_color[1]}, {dominant_color[2]})"

    # Helper function to extract texture description
    def get_texture_description(gray_image):
        lbp = local_binary_pattern(gray_image, LBP_POINTS, LBP_RADIUS, method="uniform")
        hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, LBP_POINTS + 3), range=(0, LBP_POINTS + 2))
        hist = hist.astype("float")
        hist /= (hist.sum() + 1e-6)

        uniformity = np.max(hist)
        if uniformity > 0.3:
            return "Smooth or clayey"
        elif uniformity > 0.15:
            return "Moderately textured"
        else:
            return "Coarse or gritty"

    # Class-based drainage logic
    def infer_drainage_by_soil_type(soil_type):
        mapping = {
            "Alluvial soil": "Moderately well-drained",
            "Black Soil": "High water retention",
            "Clay Soil": "Varying (sandy to clayey)",
            "Red Soil": "Generally well-drained"
        }
        return mapping.get(soil_type, "Unknown")

    # Class-based fertility logic
    def infer_fertility_by_soil_type(soil_type):
        mapping = {
            "Alluvial soil": "Moderately fertile",
            "Black Soil": "High fertility",
            "Clay Soil": "Varies",
            "Red Soil": "Low natural fertility"
        }
        return mapping.get(soil_type, "Unknown")

    # Main function to process dataset
    def process_soil_dataset(csv_path, image_folder, output_csv_path):
        df = pd.read_csv(csv_path)
        results = []

        for _, row in df.iterrows():
            image_id = row['image_id']
            soil_type = row['soil_type']
            image_path = os.path.join(image_folder, image_id)

            if not os.path.exists(image_path):
                print(f"Image not found: {image_path}")
                continue

            image = cv2.imread(image_path)
            if image is None:
                print(f"Could not read image: {image_path}")
                continue

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            color = get_dominant_color(image)
            texture = get_texture_description(gray)
            drainage = infer_drainage_by_soil_type(soil_type)
            fertility = infer_fertility_by_soil_type(soil_type)

            results.append({
                "image_id": image_id,
                "soil_type": soil_type,
                "color": color,
                "texture": texture,
                "drainage": drainage,
                "fertility": fertility
            })

        result_df = pd.DataFrame(results)
        result_df.to_csv(output_csv_path, index=False)
        return result_df

    # Example usage
    csv_input_path = "/content/drive/MyDrive/soil-classification/soil_classification-2025/train_labels.csv"
    image_folder_path = "/content/drive/MyDrive/soil-classification/soil_classification-2025/train/"
    csv_output_path = "soil_attributes_output.csv"

    # Run this after setting the correct paths
    process_soil_dataset(csv_input_path, image_folder_path, csv_output_path)
    return 0

preprocessing()
