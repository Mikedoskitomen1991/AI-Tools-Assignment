# AI Assignment - Jupyter Notebook Format

# Part 1: Theoretical Understanding

## Q1: TensorFlow vs. PyTorch
"""
- TensorFlow: static/dynamic graphs, better for production.
- PyTorch: dynamic graphs, better for research and debugging.
"""

## Q2: Jupyter Use Cases
"""
1. Interactive prototyping of models.
2. Combining code, markdown, and visualizations for presentations.
"""

## Q3: spaCy vs. String Ops
"""
spaCy provides pre-trained models for NER, POS, dependency parsing.
More accurate and efficient than regex or string methods.
"""

## Scikit-learn vs TensorFlow
"""
| Feature            | Scikit-learn          | TensorFlow          |
|--------------------|------------------------|----------------------|
| Focus              | Classical ML           | Deep Learning        |
| Beginner Friendly  | Yes                    | Medium               |
| Community Support  | Large                  | Very Large (Google)  |
"""

# Part 2: Practical Implementation

## Task 1: Iris Dataset with Scikit-learn

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Load data
df = pd.DataFrame(load_iris().data, columns=load_iris().feature_names)
df['species'] = load_iris().target

# Handle missing values
df.fillna(df.mean(), inplace=True)

# Train/test split
X = df.iloc[:, :-1]
y = df['species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, average='macro'))
print("Recall:", recall_score(y_test, y_pred, average='macro'))

## Task 2: CNN on MNIST using TensorFlow

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train[..., np.newaxis] / 255.0
x_test = x_test[..., np.newaxis] / 255.0

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    MaxPooling2D(),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5, validation_split=0.1)

print("Test Accuracy:", model.evaluate(x_test, y_test)[1])

# Visualize 5 predictions
preds = model.predict(x_test[:5])
for i in range(5):
    plt.imshow(x_test[i].squeeze(), cmap='gray')
    plt.title(f"Predicted: {np.argmax(preds[i])}")
    plt.axis('off')
    plt.show()

## Task 3: NLP with spaCy

import spacy
from textblob import TextBlob

nlp = spacy.load("en_core_web_sm")
text = "I love my new Sony headphones! Apple’s AirPods were good, but this is better."
doc = nlp(text)

print("Named Entities:")
for ent in doc.ents:
    if ent.label_ in ['PRODUCT', 'ORG']:
        print(ent.text, ent.label_)

blob = TextBlob(text)
print("\nSentiment Polarity:", blob.sentiment.polarity)
print("Sentiment:", "Positive" if blob.sentiment.polarity > 0 else "Negative")

# Part 3: Ethics & Optimization

## Ethical Considerations
"""
MNIST bias: fails on atypical handwriting
Amazon Reviews: sentiment bias from cultural language use
Mitigation: TensorFlow Fairness Indicators, spaCy custom rules
"""

## Debugging TensorFlow Code
"""
# Bug fix:
# Incorrect loss: categorical_crossentropy (with sparse labels)
# Fix:
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
"""
