import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from keras.models import load_model
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix, roc_curve, auc


# Define paths
MODEL_PATH = 'model.h5'

# Define parameters
IMG_WIDTH = 256
IMG_HEIGHT = 117
TEST_DIRS = {'D:/.DATASETS/Spectrograms/test/drone': 0, 'D:/.DATASETS/Spectrograms/test/no drone': 1}
LABELS = ["drone", "no drone"]

# Load the trained model
model = load_model(MODEL_PATH)

# Lists for predictions and true labels
total_predictions = []
total_true_labels = []
for test_dir, true_label in TEST_DIRS.items():
    print(f"Predictions for images in {test_dir}:")
    for image_name in os.listdir(test_dir):
        image_path = os.path.join(test_dir, image_name)
        img = cv2.imread(image_path)
        img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
        img = img / 255.0
        img = np.expand_dims(img, axis=0)

        prediction = model.predict(img)
        predicted_label = np.argmax(prediction)

        total_predictions.append(predicted_label)
        total_true_labels.append(true_label)

        predicted_class = LABELS[predicted_label]
        print(f"Image: {image_name}, Predicted class: {predicted_class}")


total_predictions = np.array(total_predictions)
total_true_labels = np.array(total_true_labels)

# Calculate accuracy, precision, recall, and F1 score
accuracy = accuracy_score(total_true_labels, total_predictions)
precision = precision_score(total_true_labels, total_predictions)
recall = recall_score(total_true_labels, total_predictions)
f1 = f1_score(total_true_labels, total_predictions)

print(f"\nEvaluation Metrics on Test data:")
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")


# Confusion Matrix
conf_matrix = confusion_matrix(total_true_labels, total_predictions)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=LABELS, yticklabels=LABELS)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()

# ROC Curve and AUC
fpr, tpr, _ = roc_curve(total_true_labels, total_predictions)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()