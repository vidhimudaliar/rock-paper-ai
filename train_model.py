import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import cv2
import os
from sklearn.utils import shuffle

# Paths
IMG_SAVE_PATH = 'dataset'

# Class mapping
CLASS_MAP = {
    "rock": 0,
    "paper": 1,
    "scissors": 2,
    "none": 3  # The "none" gesture
}
NUM_CLASSES = len(CLASS_MAP)

# Function to map labels
def mapper(val):
    return CLASS_MAP[val]

# Define CNN model
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
        Dropout(0.5),  # Prevent overfitting
        Dense(NUM_CLASSES, activation='softmax')
    ])
    return model

# Load images
dataset = []
for directory in os.listdir(IMG_SAVE_PATH):
    path = os.path.join(IMG_SAVE_PATH, directory)
    if not os.path.isdir(path):
        continue
    for item in os.listdir(path):
        if item.startswith("."):
            continue
        img = cv2.imread(os.path.join(path, item))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (227, 227))
        dataset.append([img, directory])

# Shuffle dataset for better training
dataset = shuffle(dataset, random_state=42)

# Split into data and labels
data, labels = zip(*dataset)
labels = list(map(mapper, labels))

# One-hot encoding
labels = tf.keras.utils.to_categorical(labels, NUM_CLASSES)

# Convert to numpy arrays
data = np.array(data, dtype="float32") / 255.0  # Normalize images

# Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)

# Model Compilation
model = get_model()
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),  # Better optimizer
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train model
model.fit(datagen.flow(data, labels, batch_size=32), epochs=15)  # Train for more epochs

# Save trained model
model.save("rock-paper-scissors-model.h5")
print("Training complete and model saved.")
