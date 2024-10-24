# Import necessary libraries
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
import os

# Function to extract frames from video
def extract_frames(video_path, num_frames=30):
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    while len(frames) < num_frames and cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (224, 224))
            frames.append(frame)
        else:
            break
    cap.release()
    return np.array(frames)

# Example of loading videos and labels
def load_video_data(video_dir, labels, num_frames=30):
    data = []
    target = []
    
    for label, videos in labels.items():
        for video in videos:
            video_path = os.path.join(video_dir, video)
            frames = extract_frames(video_path, num_frames)
            data.append(frames)
            target.append(label)
    
    return np.array(data), np.array(target)

# Define CNN-LSTM model for video classification
def build_video_classification_model(input_shape, num_classes):
    model = models.Sequential()
    
    # CNN layers for spatial feature extraction
    model.add(layers.TimeDistributed(layers.Conv2D(32, (3, 3), activation='relu'), input_shape=input_shape))
    model.add(layers.TimeDistributed(layers.MaxPooling2D((2, 2))))
    model.add(layers.TimeDistributed(layers.Conv2D(64, (3, 3), activation='relu')))
    model.add(layers.TimeDistributed(layers.MaxPooling2D((2, 2))))
    
    # Flatten the CNN output
    model.add(layers.TimeDistributed(layers.Flatten()))
    
    # LSTM for temporal analysis
    model.add(layers.LSTM(64, return_sequences=False))
    
    # Dense layers for classification
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(num_classes, activation='softmax'))
    
    return model

# Example labeled video data
video_directory = "path_to_video_folder"
labels = {
    0: ["video1.mp4", "video2.mp4"],  # Class 0
    1: ["video3.mp4", "video4.mp4"],  # Class 1
}

# Load video data
X, y = load_video_data(video_directory, labels)
y = to_categorical(y, num_classes=2)

# Build and compile the model
input_shape = (30, 224, 224, 3)  # (num_frames, height, width, channels)
model = build_video_classification_model(input_shape=input_shape, num_classes=2)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model (for demonstration purposes)
model.fit(X, y, epochs=10, batch_size=2)

# Model evaluation
test_video_path = "path_to_test_video.mp4"
test_frames = extract_frames(test_video_path)
test_frames = np.expand_dims(test_frames, axis=0)  # Add batch dimension

predictions = model.predict(test_frames)
predicted_class = np.argmax(predictions[0])
print(f"Predicted class: {predicted_class}")
