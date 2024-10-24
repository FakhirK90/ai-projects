# Import necessary libraries
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import imdb

# Load the IMDb dataset (keeping the top 10,000 most frequent words)
vocab_size = 10000
max_length = 100
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=vocab_size)

# Pad the sequences to ensure uniform input length
X_train_padded = pad_sequences(X_train, maxlen=max_length, padding='post')
X_test_padded = pad_sequences(X_test, maxlen=max_length, padding='post')

# Define the LSTM-based sentiment analysis model
embedding_dim = 128
model = models.Sequential([
    layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length),
    layers.LSTM(128),
    layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train_padded, y_train, epochs=5, validation_data=(X_test_padded, y_test), batch_size=64)

# Evaluate the model
loss, accuracy = model.evaluate(X_test_padded, y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Plot accuracy and loss over time
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Test Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Test Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Function to predict the sentiment of new text
def predict_sentiment(text):
    # Tokenize and pad the input text
    tokenizer = Tokenizer(num_words=vocab_size)
    tokenizer.fit_on_texts([text])
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=max_length, padding='post')
    
    # Predict the sentiment
    prediction = model.predict(padded_sequence)
    sentiment = 'Positive' if prediction > 0.5 else 'Negative'
    return sentiment

# Test the predict_sentiment function with a new review
sample_review = "The movie was fantastic and had a great storyline!"
predicted_sentiment = predict_sentiment(sample_review)
print(f"Predicted Sentiment: {predicted_sentiment}")
