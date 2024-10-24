# Import necessary libraries
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Define a function to preprocess the data
def preprocess_data(conversations, num_words=10000, max_length=20):
    input_texts, target_texts = zip(*conversations)
    input_tokenizer = Tokenizer(num_words=num_words)
    target_tokenizer = Tokenizer(num_words=num_words)
    
    input_tokenizer.fit_on_texts(input_texts)
    target_tokenizer.fit_on_texts(target_texts)
    
    input_sequences = input_tokenizer.texts_to_sequences(input_texts)
    target_sequences = target_tokenizer.texts_to_sequences(target_texts)
    
    input_sequences = pad_sequences(input_sequences, maxlen=max_length, padding='post')
    target_sequences = pad_sequences(target_sequences, maxlen=max_length, padding='post')
    
    return input_sequences, target_sequences, input_tokenizer, target_tokenizer

# Define the Seq2Seq model
def build_seq2seq_model(input_vocab_size, target_vocab_size, embedding_dim=256, lstm_units=512):
    # Encoder
    encoder_inputs = layers.Input(shape=(None,))
    encoder_embedding = layers.Embedding(input_vocab_size, embedding_dim)(encoder_inputs)
    encoder_lstm = layers.LSTM(lstm_units, return_sequences=True, return_state=True)
    encoder_outputs, state_h, state_c = encoder_lstm(encoder_embedding)
    
    # Decoder
    decoder_inputs = layers.Input(shape=(None,))
    decoder_embedding = layers.Embedding(target_vocab_size, embedding_dim)(decoder_inputs)
    decoder_lstm = layers.LSTM(lstm_units, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=[state_h, state_c])
    
    # Output layer
    outputs = layers.Dense(target_vocab_size, activation='softmax')(decoder_outputs)
    
    # Define the model
    model = models.Model([encoder_inputs, decoder_inputs], outputs)
    return model

# Example conversations data (input-output pairs)
conversations = [
    ("Hi there!", "Hello! How can I help you?"),
    ("What is your name?", "I am a chatbot created to assist you."),
    ("How are you?", "I'm just a program, but thanks for asking!"),
    ("Tell me a joke.", "Why did the scarecrow win an award? Because he was outstanding in his field!"),
]

# Preprocess the data
input_sequences, target_sequences, input_tokenizer, target_tokenizer = preprocess_data(conversations)

# Build and compile the Seq2Seq model
model = build_seq2seq_model(input_vocab_size=len(input_tokenizer.word_index) + 1, target_vocab_size=len(target_tokenizer.word_index) + 1)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model (this is a simplified example)
model.fit([input_sequences, target_sequences], target_sequences, epochs=10, batch_size=64)

# Function to generate responses from the chatbot
def generate_response(input_text, input_tokenizer, target_tokenizer, model, max_length=20):
    input_seq = input_tokenizer.texts_to_sequences([input_text])
    input_seq = pad_sequences(input_seq, maxlen=max_length, padding='post')
    
    # Start generating the response
    response = []
    for _ in range(max_length):
        target_seq = pad_sequences([response], maxlen=max_length, padding='post')
        prediction = model.predict([input_seq, target_seq])
        predicted_word_index = np.argmax(prediction[0, -1, :])
        predicted_word = target_tokenizer.index_word.get(predicted_word_index, '')
        
        if predicted_word == '<end>':
            break
        
        response.append(predicted_word)
    
    return ' '.join(response)

# Test the generate_response function with user input
user_input = "Hi there!"
chatbot_response = generate_response(user_input, input_tokenizer, target_tokenizer, model)
print(f"Chatbot Response: {chatbot_response}")
