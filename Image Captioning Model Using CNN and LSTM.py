# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Define a simple CNN model for feature extraction
def build_cnn_model():
    base_model = tf.keras.applications.VGG16(weights='imagenet', include_top=False, pooling='avg')
    model = models.Model(inputs=base_model.input, outputs=base_model.output)
    return model

# Prepare the data (example data structure)
def preprocess_data(image_paths, captions, num_words=10000, max_length=20):
    # Tokenize the captions
    tokenizer = Tokenizer(num_words=num_words)
    tokenizer.fit_on_texts(captions)
    sequences = tokenizer.texts_to_sequences(captions)
    
    # Pad the sequences
    padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post')
    
    return image_paths, padded_sequences, tokenizer

# Build the LSTM model for caption generation
def build_lstm_model(vocab_size, embedding_dim=256, lstm_units=512):
    inputs = layers.Input(shape=(None,))
    x = layers.Embedding(vocab_size, embedding_dim)(inputs)
    x = layers.LSTM(lstm_units)(x)
    outputs = layers.Dense(vocab_size, activation='softmax')(x)
    
    model = models.Model(inputs, outputs)
    return model

# Example data (image paths and corresponding captions)
image_paths = ['image1.jpg', 'image2.jpg']  # Placeholder paths
captions = ["A dog playing in the park", "A cat sitting on a chair"]

# Preprocess the data
image_paths, padded_sequences, tokenizer = preprocess_data(image_paths, captions)

# Build the models
cnn_model = build_cnn_model()
lstm_model = build_lstm_model(vocab_size=len(tokenizer.word_index) + 1)

# Compile the LSTM model
lstm_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Placeholder: Train the LSTM model (this is a simplified example)
# Here, you would extract features using cnn_model and use them to train the lstm_model
# features = cnn_model.predict(image_data)  # Extract features from images
# lstm_model.fit(features, padded_sequences, epochs=10, batch_size=64)

# Function to generate captions for new images
def generate_caption(image_path, tokenizer, cnn_model, lstm_model, max_length=20):
    # Load and preprocess the image
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Extract features from the image
    features = cnn_model.predict(img_array)
    
    # Start generating the caption
    caption = []
    for _ in range(max_length):
        sequence = pad_sequences([tokenizer.texts_to_sequences(caption)], maxlen=max_length)
        prediction = lstm_model.predict(sequence)
        predicted_word_index = np.argmax(prediction)
        predicted_word = tokenizer.index_word.get(predicted_word_index, '')
        
        if predicted_word == '<end>':
            break
        
        caption.append(predicted_word)
    
    return ' '.join(caption)

# Test the generate_caption function with a new image
sample_image_path = 'image1.jpg'  # Placeholder for an actual image path
generated_caption = generate_caption(sample_image_path, tokenizer, cnn_model, lstm_model)
print(f"Generated Caption: {generated_caption}")
