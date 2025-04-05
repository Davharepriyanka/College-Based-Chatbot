import nltk
from nltk.stem import WordNetLemmatizer
import json
import pickle
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD
import random

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Initialize lists to hold words, classes, and documents
words = []
classes = []
documents = []
ignore_words = ['?', '!']

# Load intents from JSON file
data_file = open('data.json').read()
intents = json.loads(data_file)

# Process intents to build vocabulary
for intent in intents['intents']:
    for pattern in intent['patterns']:
        # Tokenize each word
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        # Add documents in the corpus
        documents.append((w, intent['tag']))

        # Add to our classes list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Lemmatize and lower each word, then remove duplicates
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))

# Sort classes
classes = sorted(list(set(classes)))

# Print statistics
print(len(documents), "documents")
print(len(classes), "classes", classes)
print(len(words), "unique lemmatized words", words)

# Save the words and classes for later use
pickle.dump(words, open('texts.pkl', 'wb'))
pickle.dump(classes, open('labels.pkl', 'wb'))

# Create our training data
training = []
# Create an empty array for our output
output_empty = [0] * len(classes)

# Training set, bag of words for each sentence
for doc in documents:
    # Initialize our bag of words
    bag = []
    # List of tokenized words for the pattern
    pattern_words = doc[0]
    # Lemmatize each word - create base word
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    
    # Create our bag of words array with 1 if word match found in current pattern
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)

    # Output is a '0' for each tag and '1' for current tag (for each pattern)
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    
    # Ensure that the bag and output_row are of correct lengths
    if len(bag) == len(words) and len(output_row) == len(classes):
        training.append([bag, output_row])
    else:
        print(f"Mismatch found - Bag length: {len(bag)}, Output row length: {len(output_row)}")

# Shuffle our features and turn into np.array
random.shuffle(training)

# Convert to np.array 
