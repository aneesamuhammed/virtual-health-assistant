import json
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
import pickle
import random

# Ensure necessary NLTK data is available
nltk.download('punkt')
nltk.download('wordnet')

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Load the dataset
with open("1.json", "r") as file:
    data = json.load(file)

# Preprocessing
words = []
labels = []
documents = []
ignore_words = ["?", "!", ".", ","]

for entry in data:
    question = entry["question"]
    tag = entry["tags"][0]  # Assuming only one tag per question
    
    word_list = nltk.word_tokenize(question)
    words.extend(word_list)
    documents.append((word_list, tag))
    
    if tag not in labels:
        labels.append(tag)

# Lemmatize, remove duplicates, and sort words
words = sorted(set([lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]))
labels = sorted(labels)

# Ensure input feature size is exactly 389
required_input_size = 389
while len(words) < required_input_size:
    words.append("placeholder_word")  # Add dummy words if fewer
words = words[:required_input_size]  # Trim excess words

# Create training data
training = []
output_empty = [0] * len(labels)

for doc in documents:
    bag = [1 if w in [lemmatizer.lemmatize(word.lower()) for word in doc[0]] else 0 for w in words]
    output_row = list(output_empty)
    output_row[labels.index(doc[1])] = 1
    training.append([bag, output_row])

# Shuffle and convert to numpy array
random.shuffle(training)
training = np.array(training, dtype=object)

train_x = np.array(list(training[:, 0]))
train_y = np.array(list(training[:, 1]))

# Build the model with updated input size (389)
model = Sequential([
    Dense(128, input_shape=(389,), activation='relu'),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(len(labels), activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.01), metrics=['accuracy'])

# Train the model
model.fit(train_x, train_y, epochs=200, batch_size=8, verbose=1)

# Save model and data
model.save("chatbot_model.hdf5")

with open("words.pkl", "wb") as f:
    pickle.dump(words, f)

with open("labels.pkl", "wb") as f:
    pickle.dump(labels, f)

print("Training complete. Model saved.")
