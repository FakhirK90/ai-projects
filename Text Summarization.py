# Import necessary libraries
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

# Sample data (for real use, you would need a large summarization dataset)
texts = [
    "The quick brown fox jumps over the lazy dog. It is a common sentence used to test typing speed.",
    "Artificial Intelligence is transforming the world in various domains, from healthcare to finance."
]
summaries = ["The fox jumps over the dog.", "AI is transforming many domains."]

# Tokenization and padding for text and summaries
max_text_length = 50
max_summary_length = 10
vocab_size = 10000

tokenizer = Tokenizer(num_words=vocab_size)
tokenizer.fit_on_texts(texts + summaries)

text_sequences = tokenizer.texts_to_sequences(texts)
summary_sequences = tokenizer.texts_to_sequences(summaries)

text_padded = pad_sequences(text_sequences, maxlen=max_text_length, padding='post')
summary_padded = pad_sequences(summary_sequences, maxlen=max_summary_length, padding='post')

# Define the Seq2Seq model with Attention
embedding_dim = 128
latent_dim = 256

# Encoder
encoder_inputs = layers.Input(shape=(max_text_length,))
encoder_embedding = layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim)(encoder_inputs)
encoder_lstm, state_h, state_c = layers.LSTM(latent_dim, return_state=True)(encoder_embedding)

# Decoder
decoder_inputs = layers.Input(shape=(max_summary_length,))
decoder_embedding = layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim)(decoder_inputs)
decoder_lstm = layers.LSTM(latent_dim, return_sequences=True, return_state=False)(decoder_embedding, initial_state=[state_h, state_c])

# Attention mechanism
attention = layers.Attention()([decoder_lstm, encoder_lstm])
decoder_concat_input = layers.Concatenate(axis=-1)([decoder_lstm, attention])

# Dense layer to generate word probabilities
output_layer = layers.Dense(vocab_size, activation='softmax')(decoder_concat_input)

# Define the full model
model = models.Model([encoder_inputs, decoder_inputs], output_layer)

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit([text_padded, summary_padded], np.expand_dims(summary_padded, -1), epochs=10, batch_size=64)

# Function to summarize new text
def summarize_text(input_text):
    input_seq = tokenizer.texts_to_sequences([input_text])
    input_padded = pad_sequences(input_seq, maxlen=max_text_length, padding='post')
    
    # Encode the input sequence
    encoder_output, state_h, state_c = encoder_lstm.predict(input_padded)
    
    # Start with the <START> token
    summary = ['<START>']
    summary_seq = tokenizer.texts_to_sequences(summary)
    summary_padded = pad_sequences(summary_seq, maxlen=max_summary_length, padding='post')
    
    # Generate summary word by word
    for _ in range(max_summary_length):
        decoder_output = model.predict([input_padded, summary_padded])
        next_word_id = np.argmax(decoder_output[0, -1, :])
        next_word = tokenizer.index_word.get(next_word_id, '<UNK>')
        
        # If we reach the <END> token, stop generating
        if next_word == '<END>':
            break
        
        summary.append(next_word)
        summary_seq = tokenizer.texts_to_sequences([summary])
        summary_padded = pad_sequences(summary_seq, maxlen=max_summary_length, padding='post')
    
    return ' '.join(summary[1:])

# Test the summarize_text function
new_text = "Artificial Intelligence is revolutionizing industries with automation and smart technologies."
predicted_summary = summarize_text(new_text)
print(f"Predicted Summary: {predicted_summary}")
