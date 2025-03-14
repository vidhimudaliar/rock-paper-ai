import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import cv2
import os

# Path to the image dataset (generated by collect_data.py)
IMG_SAVE_PATH = 'dataset'

# Class mapping for gestures
CLASS_MAP = {
    "rock": 0,
    "paper": 1,
    "scissors": 2,
    "none": 3  # For "none" gesture, in case it's used
}

NUM_CLASSES = len(CLASS_MAP)

# Mapper function for labels
def mapper(val):
    return CLASS_MAP[val]

# Function to define the CNN model architecture
def get_model():
    model = Sequential([
        Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(227, 227, 3)),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(512, activation='relu'),
        Dense(NUM_CLASSES, activation='softmax')
    ])
    return model

# Load images from the directory (generated by collect_data.py)
dataset = []

# Loop through directories (rock, paper, scissors)
for directory in os.listdir(IMG_SAVE_PATH):
    path = os.path.join(IMG_SAVE_PATH, directory)
    if not os.path.isdir(path):
        continue
    for item in os.listdir(path):
        # Skip hidden files (like .DS_Store)
        if item.startswith("."):
            continue
        img = cv2.imread(os.path.join(path, item))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        img = cv2.resize(img, (227, 227))  # Resize image to 227x227
        dataset.append([img, directory])  # Append image and its label

# Split dataset into images and labels
data, labels = zip(*dataset)
labels = list(map(mapper, labels))  # Map labels to numerical values

# One-hot encode the labels (rock -> [1,0,0], paper -> [0,1,0], etc.)
labels = tf.keras.utils.to_categorical(labels, NUM_CLASSES)  # One-hot encoding

# Convert data into a numpy array for model input
data = np.array(data)

# Create ImageDataGenerator for data augmentation (optional but useful)
datagen = ImageDataGenerator(rescale=1.0/255.0)  # Normalize pixel values

# Define the model
model = get_model()
model.compile(
    optimizer='sgd',  # Using SGD (Stochastic Gradient Descent) optimizer
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train the model with data augmentation
model.fit(datagen.flow(data, labels, batch_size=32), epochs=10)

# Save the trained model for later use
model.save("rock-paper-scissors-model.h5")

print("Training complete and model saved.")
