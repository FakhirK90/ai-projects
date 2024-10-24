# Import necessary libraries
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Define tokenization and preprocessing functions
def preprocess_data(sentence_pairs, num_words=10000, max_len=20):
    input_texts, target_texts = zip(*sentence_pairs)
    input_tokenizer = Tokenizer(num_words=num_words)
    target_tokenizer = Tokenizer(num_words=num_words)
    
    input_tokenizer.fit_on_texts(input_texts)
    target_tokenizer.fit_on_texts(target_texts)
    
    input_sequences = input_tokenizer.texts_to_sequences(input_texts)
    target_sequences = target_tokenizer.texts_to_sequences(target_texts)
    
    input_sequences = pad_sequences(input_sequences, maxlen=max_len, padding='post')
    target_sequences = pad_sequences(target_sequences, maxlen=max_len, padding='post')
    
    return input_sequences, target_sequences, input_tokenizer, target_tokenizer

# Define the Seq2Seq model with attention mechanism
def build_seq2seq_model(input_vocab_size, target_vocab_size, embedding_dim=256, lstm_units=512):
    # Encoder
    encoder_inputs = layers.Input(shape=(None,))
    encoder_embedding = layers.Embedding(input_vocab_size, embedding_dim)(encoder_inputs)
    encoder_lstm = layers.LSTM(lstm_units, return_sequences=True, return_state=True)
    encoder_outputs, state_h, state_c = encoder_lstm(encoder_embedding)
    
    # Attention mechanism
    attention = layers.AdditiveAttention()([encoder_outputs, encoder_outputs])
    
    # Decoder
    decoder_inputs = layers.Input(shape=(None,))
    decoder_embedding = layers.Embedding(target_vocab_size, embedding_dim)(decoder_inputs)
    decoder_lstm = layers.LSTM(lstm_units, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=[state_h, state_c])
    
    # Combine decoder output with attention
    attention_output = layers.Concatenate(axis=-1)([decoder_outputs, attention])
    
    # Output layer
    decoder_dense = layers.Dense(target_vocab_size, activation='softmax')
    outputs = decoder_dense(attention_output)
    
    # Define the model
    model = models.Model([encoder_inputs, decoder_inputs], outputs)
    return model

# Sample sentence pairs (English to French)
sentence_pairs = [
    ("I love you", "Je t'aime"),
    ("How are you?", "Comment ça va?"),
    ("What is your name?", "Quel est votre nom?")
]

# Preprocess the data
input_sequences, target_sequences, input_tokenizer, target_tokenizer = preprocess_data(sentence_pairs)

# Build and compile the Seq2Seq model
model = build_seq2seq_model(input_vocab_size=10000, target_vocab_size=10000)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model (this is a simplified example)
model.fit([input_sequences, target_sequences], target_sequences, epochs=10, batch_size=64)

# Function to translate new sentences
def translate_sentence(sentence, input_tokenizer, target_tokenizer, model, max_len=20):
    input_seq = input_tokenizer.texts_to_sequences([sentence])
    input_seq = pad_sequences(input_seq, maxlen=max_len, padding='post')
    
    # Predict the target sequence
    predictions = model.predict([input_seq, input_seq])
    predicted_seq = np.argmax(predictions[0], axis=-1)
    
    # Convert predicted sequence to text
    translated_sentence = ' '.join([target_tokenizer.index_word.get(idx, '') for idx in predicted_seq if idx > 0])
    return translated_sentence

# Test the translation function with a new sentence
sample_sentence = "I love you"
translated_sentence = translate_sentence(sample_sentence, input_tokenizer, target_tokenizer, model)
print(f"Translated Sentence: {translated_sentence}")
