import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

DATASET_PATH = "D:/.DATASETS/Spectrograms/train"
LABELS = ["drone", "no drone"]

# Function to enhance contrast using histogram equalization
def enhance_contrast(image):
    img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])  # Histogram equalization on the luminance channel
    img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    return img_output

# Function to enhance edges using Laplacian filter
def enhance_edges(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    laplacian = np.uint8(np.absolute(laplacian))
    return laplacian

# Process each image in the dataset
for label in LABELS:
    folder_path = os.path.join(DATASET_PATH, label)
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        image = cv2.imread(file_path)
        if image is not None:
            image = cv2.resize(image, (256, 117))  # Resize to consistent dimensions
            contrast_image = enhance_contrast(image)
            edges_image = enhance_edges(image)

            # Plot original, contrast enhanced, and edge enhanced images
            plt.figure(figsize=(12, 4))
            plt.subplot(1, 3, 1)
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            plt.title('Original Image')
            plt.axis('off')

            plt.subplot(1, 3, 2)
            plt.imshow(cv2.cvtColor(contrast_image, cv2.COLOR_BGR2RGB))
            plt.title('Contrast Enhanced')
            plt.axis('off')

            plt.subplot(1, 3, 3)
            plt.imshow(edges_image, cmap='gray')
            plt.title('Edge Enhanced')
            plt.axis('off')

            plt.show()
