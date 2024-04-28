import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import LSTM, Embedding, Dense, Dropout, Flatten
import numpy as np

data = open("data.txt", "r+").readlines()
x, y = [], []

def generate_text(model, tokenizer, seed_text, next_words=50, max_sequence_len=20):
    """
    Generate text using a trained Keras model.
    
    Parameters:
        model (tf.keras.Model): The trained Keras model.
        tokenizer (Tokenizer): Tokenizer used to tokenize the text.
        seed_text (str): The seed text to start text generation.
        next_words (int): Number of words to generate.
        max_sequence_len (int): Maximum length of sequences to feed into the model.
    
    Returns:
        str: Generated text.
    """
    generated_text = seed_text
    for _ in range(next_words):
        # Tokenize the seed text
        token_list = tokenizer.texts_to_sequences([generated_text])[0]
        # Pad the sequences to the max_sequence_len
        token_list = pad_sequences([token_list], maxlen=max_sequence_len, padding='pre')
        # Predict the next word
        predicted = model.predict(token_list, verbose=0).argmax()
        # Convert the predicted index to word
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        generated_text += " " + output_word
    return generated_text

def unique(a):
	b = []
	for i in a:
		if i not in b:
			b.append(i)
	return b

for i in data:
	if len(i) > 1:
		buffer = []
		for o in range(1, len(i)):
			buffer.append(i[:o])
			x.append(buffer)
			y.append(i[o])

del buffer


#max([len(i)for i in x])

tokenizer = Tokenizer()
tokenizer.fit_on_texts(data)

maxlen = len(tokenizer.word_index) + 1

x, y = tokenizer.texts_to_sequences(x), tokenizer.texts_to_sequences(y)
x, y = pad_sequences(x, maxlen=maxlen), pad_sequences(y, maxlen=maxlen)

print(maxlen)
model = tf.keras.models.Sequential([
	Embedding(maxlen, 32),
	LSTM(128),
	Dense(100, activation='relu'),
	Dense(maxlen, activation="softmax")

])

model.summary()
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])


model.fit(x, y, epochs=100, batch_size=1024)

while True:
	user = input("vous : ")
	print(generate_text(model, tokenizer, user, len(user), maxlen))
