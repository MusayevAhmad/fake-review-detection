"""
TF-IDF with N-grams and Logistic Regression for Fake Review Detection

This script implements a traditional machine learning approach using:
- TF-IDF vectorization with unigrams, bigrams, and trigrams
- Logistic regression classifier
- Cross-validation and grid search for optimization

Author: Group 86
Date: 2024-2025
Course: VU Machine Learning
"""

import pandas as pd
import nltk
from nltk.util import ngrams
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
import numpy as np
import os
import joblib

# Download necessary NLTK resources if not already downloaded
print("Checking and downloading required NLTK resources...")
nltk_resources = ['punkt', 'stopwords', 'wordnet', 'omw-1.4']
for resource in nltk_resources:
    try:
        nltk.data.find(f'corpora/{resource}')
        print(f"Resource '{resource}' is already downloaded.")
    except LookupError:
        print(f"Downloading '{resource}'...")
        nltk.download(resource)

# Load dataset
file_path = '/Users/ahmedmusayev/Desktop/VU/Year 2/P4/ML/MachineLearningProject/datasets/fake_reviews_dataset.csv'
df = pd.read_csv(file_path, encoding="utf-8")

# Define stopwords set
stop_words = set(stopwords.words('english'))

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Tokenization function that removes stopwords and applies lemmatization
def tokenize_text(text):
    if isinstance(text, str):
        # Lowercase & tokenize
        tokens = word_tokenize(text.lower())
        
        # Keep only alphabetic tokens, remove stopwords, and apply lemmatization
        tokens = [lemmatizer.lemmatize(word) for word in tokens 
                 if word.isalpha() and word not in stop_words]
        
        return tokens
    return []

# Apply tokenization
print("Tokenizing and lemmatizing text...")
df["tokens"] = df["text_"].apply(tokenize_text)

# Function to extract bigrams & trigrams
def get_common_ngrams(text_list, n=2, top_n=10):
    """
    text_list: List of token lists, e.g., df["tokens"].tolist()
    n: the n-gram size (2 for bigrams, 3 for trigrams, etc.)
    top_n: how many most common n-grams to return
    """
    all_ngrams = [ngram for tokens in text_list for ngram in ngrams(tokens, n)]
    return Counter(all_ngrams).most_common(top_n)

# Get bigrams & trigrams for CG and OR
cg_bigrams = get_common_ngrams(df[df["label"] == "CG"]["tokens"].tolist(), n=2)
cg_trigrams = get_common_ngrams(df[df["label"] == "CG"]["tokens"].tolist(), n=3)
or_bigrams = get_common_ngrams(df[df["label"] == "OR"]["tokens"].tolist(), n=2)
or_trigrams = get_common_ngrams(df[df["label"] == "OR"]["tokens"].tolist(), n=3)

# Print results
print("\n **Top 10 AI-Generated (CG) Bigrams (Stopwords Removed, Lemmatized):**")
for bigram, count in cg_bigrams:
    print(f"{bigram}: {count}")

print("\n **Top 10 AI-Generated (CG) Trigrams (Stopwords Removed, Lemmatized):**")
for trigram, count in cg_trigrams:
    print(f"{trigram}: {count}")

print("\n **Top 10 Human (OR) Bigrams (Stopwords Removed, Lemmatized):**")
for bigram, count in or_bigrams:
    print(f"{bigram}: {count}")

print("\n **Top 10 Human (OR) Trigrams (Stopwords Removed, Lemmatized):**")
for trigram, count in or_trigrams:
    print(f"{trigram}: {count}")

# Split the data into training, validation, and test sets (80/10/10)
print("\nSplitting data into training, validation, and test sets...")
train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['label'])

print(f"Training set size: {train_df.shape[0]} ({train_df.shape[0]/df.shape[0]:.2%})")
print(f"Validation set size: {val_df.shape[0]} ({val_df.shape[0]/df.shape[0]:.2%})")
print(f"Test set size: {test_df.shape[0]} ({test_df.shape[0]/df.shape[0]:.2%})")

# TF-IDF Vectorization with n-grams
print("\nApplying TF-IDF vectorization with unigrams, bigrams, and trigrams...")
tfidf_vectorizer = TfidfVectorizer(
    tokenizer=tokenize_text,
    max_features=10000,  # Increased from 5000
    min_df=3,            # Reduced from 5
    max_df=0.9,          # Increased from 0.8
    ngram_range=(1, 3),  # Includes unigrams, bigrams, and trigrams
    sublinear_tf=True    # Apply sublinear tf scaling (log scaling)
)

X_train = tfidf_vectorizer.fit_transform(train_df['text_'])
print(f"TF-IDF features: {X_train.shape[1]}")

# Use the best parameters found from previous grid search
# Instead of running grid search again
print("\nCreating model with best parameters from previous grid search...")
best_params = {
    'C': 10.0,
    'max_iter': 1000,
    'class_weight': 'balanced',
    'penalty': 'l2',
    'solver': 'liblinear'
}

print(f"Using parameters: {best_params}")
best_log_reg = LogisticRegression(**best_params)

# Train the model with the best parameters
print("\nTraining model on training set...")
best_log_reg.fit(X_train, train_df['label'])
print("Model training complete!")

# Create directories for saving models and results
models_dir = 'ML_models/saved_models'
results_dir = 'results'
os.makedirs(models_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)

# Save the best model and vectorizer for later use
print("\nSaving trained model and vectorizer...")
joblib.dump(tfidf_vectorizer, f'{models_dir}/tfidf_ngram_vectorizer.joblib')
joblib.dump(best_log_reg, f'{models_dir}/best_logistic_regression_ngram.joblib')
print(f"Models saved to {models_dir}/")

# Save training parameters to a file
with open(f'{results_dir}/tfidf_ngram_training_info.txt', 'w') as f:
    f.write("TF-IDF Vectorizer Parameters:\n")
    for param, value in tfidf_vectorizer.get_params().items():
        f.write(f"{param}: {value}\n")
    
    f.write("\nLogistic Regression Parameters:\n")
    for param, value in best_log_reg.get_params().items():
        f.write(f"{param}: {value}\n")
    
    f.write(f"\nTraining set size: {train_df.shape[0]} samples\n")
    f.write(f"Number of features: {X_train.shape[1]}\n")

print(f"\nTraining information saved to '{results_dir}/tfidf_ngram_training_info.txt'")
print("\nTF-IDF with N-grams Training Complete!")