import tensorflow as tf
from keras import layers, models
import matplotlib.pyplot as plt
import os
import cv2
from tqdm import tqdm 
import numpy as np
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.applications import DenseNet121, VGG16
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Model, load_model
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.applications import DenseNet121, InceptionV3
from keras.optimizers import Adam
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
from keras import layers, models, Input
from tqdm import tqdm
import cv2
from sklearn.utils import shuffle
import os
import datetime
import random
from keras.callbacks import EarlyStopping

# Parameters
DATASET_PATH = "D:/.DATASETS/Spectrograms/train"
LABELS = ["drone", "no drone"]

epochs = 150
batch_size = 32
IMG_WIDTH = 256
IMG_HEIGHT = 117
X = []
Y = []

for dir in os.listdir(DATASET_PATH):
    print("Folder:", dir)
    for item in tqdm(os.listdir(os.path.join(DATASET_PATH, dir))):
        data = cv2.imread(os.path.join(DATASET_PATH, dir, item))
        data = cv2.resize(data, (IMG_WIDTH, IMG_HEIGHT))
        data = data / 255
        target = LABELS.index(dir) 
        X.append(data)
        Y.append(target)

trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.1, random_state=1)
trainX, trainY = shuffle(trainX, trainY, random_state=0)
trainX = np.array(trainX)
testX = np.array(testX)
trainY = to_categorical(trainY, num_classes=len(LABELS))
testY = to_categorical(testY, num_classes=len(LABELS))

print("\n")
print("trainX shape:", trainX.shape)
print("trainY shape:", trainY.shape)
print("testX shape:", testX.shape)
print("testY shape:", testY.shape)
print("\n")

#Random plot
#random_index = random.randint(0, len(trainX) - 1)
#plt.imshow(trainX[random_index])
#plt.show()


# Model pretrained
'''#base_model = VGG16(weights='imagenet', include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))
base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))

for layer in base_model.layers[:-4]:
    layer.trainable = False
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(len(LABELS), activation='softmax')(x)  

model = Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(trainX, trainY, epochs=epochs, batch_size=batch_size, validation_split=0.2, verbose=1)

model.save("model.h5")
'''

# Model custom
custom_model = Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
x = layers.Conv2D(32, (3, 3), activation='relu')(custom_model)
x = layers.MaxPooling2D((2, 2))(x)
x = layers.Conv2D(64, (3, 3), activation='relu')(x)
x = layers.MaxPooling2D((2, 2))(x)
x = layers.Conv2D(128, (3, 3), activation='relu')(x)
x = layers.MaxPooling2D((2, 2))(x)
x = layers.Conv2D(256, (3, 3), activation='relu')(x)
x = layers.MaxPooling2D((2, 2))(x)
x = layers.Conv2D(512, (3, 3), activation='relu')(x)
x = layers.MaxPooling2D((2, 2))(x)
x = layers.Flatten()(x)
x = layers.Dense(512, activation='relu')(x)
x = layers.Dropout(0.5)(x)

output_layer = layers.Dense(len(LABELS), activation='softmax')(x)
model = models.Model(inputs=custom_model, outputs=output_layer)
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, verbose=1, restore_best_weights=True)

history = model.fit(trainX, trainY, 
                    epochs=epochs, 
                    batch_size=batch_size, 
                    validation_split=0.1, 
                    verbose=1,
                    callbacks=[early_stopping]
                    )

model.save("model.h5")


# Scores
test_loss, test_acc = model.evaluate(testX, testY, verbose=2)

predictions = model.predict(testX)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = np.argmax(testY, axis=1)
precision = precision_score(true_classes, predicted_classes, average='macro')
recall = recall_score(true_classes, predicted_classes, average='macro')
f1 = f1_score(true_classes, predicted_classes, average='macro')
print(f'Test loss: {test_loss}')
print(f'Test accuracy: {test_acc}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1-Score: {f1}')

# Plots
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()
