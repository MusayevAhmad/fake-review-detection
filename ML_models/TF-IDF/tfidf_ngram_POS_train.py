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
nltk_resources = ['punkt', 'stopwords', 'wordnet', 'omw-1.4', 'averaged_perceptron_tagger']
for resource in nltk_resources:
    try:
        nltk.data.find(f'corpora/{resource}' if resource != 'averaged_perceptron_tagger' else f'taggers/{resource}')
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

# Tokenization function with POS tagging, lemmatization, and stopword removal
def tokenize_text_pos(text):
    if isinstance(text, str):
        # Lowercase & tokenize
        tokens = word_tokenize(text.lower())
        
        # Keep only alphabetic tokens and remove stopwords
        tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
        
        # POS tagging
        tagged_tokens = nltk.pos_tag(tokens)
        
        # Lemmatize with POS information and create POS-enhanced tokens
        pos_tokens = []
        for word, tag in tagged_tokens:
            # Convert Penn Treebank tags to WordNet POS tags
            if tag.startswith('J'):
                pos = 'a'  # adjective
            elif tag.startswith('V'):
                pos = 'v'  # verb
            elif tag.startswith('N'):
                pos = 'n'  # noun
            elif tag.startswith('R'):
                pos = 'r'  # adverb
            else:
                pos = 'n'  # default to noun
            
            # Lemmatize with POS
            lemma = lemmatizer.lemmatize(word, pos=pos)
            
            # Add POS-enhanced token (for important parts of speech)
            if tag.startswith('N') or tag.startswith('V') or tag.startswith('J') or tag.startswith('R'):
                pos_tokens.append(f"{lemma}_{tag[:2]}")
            else:
                pos_tokens.append(lemma)
        
        return pos_tokens
    return []

# Apply tokenization
print("Tokenizing text with POS tagging and lemmatization...")
df["tokens"] = df["text_"].apply(tokenize_text_pos)

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
print("\n **Top 10 AI-Generated (CG) Bigrams (With POS Tagging):**")
for bigram, count in cg_bigrams:
    print(f"{bigram}: {count}")

print("\n **Top 10 AI-Generated (CG) Trigrams (With POS Tagging):**")
for trigram, count in cg_trigrams:
    print(f"{trigram}: {count}")

print("\n **Top 10 Human (OR) Bigrams (With POS Tagging):**")
for bigram, count in or_bigrams:
    print(f"{bigram}: {count}")

print("\n **Top 10 Human (OR) Trigrams (With POS Tagging):**")
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
print("\nApplying TF-IDF vectorization with unigrams, bigrams, and trigrams (with POS tagging)...")
tfidf_vectorizer = TfidfVectorizer(
    tokenizer=tokenize_text_pos,
    max_features=50000,
    min_df=3,
    max_df=0.9,
    ngram_range=(1, 3),
    sublinear_tf=False
)

X_train = tfidf_vectorizer.fit_transform(train_df['text_'])
print(f"TF-IDF features: {X_train.shape[1]}")

# Use the best parameters found from previous grid search
print("\nCreating model with best parameters from previous grid search...")
best_params = {
    'C': 10.0,
    'max_iter': 5000,
    'class_weight': 'balanced',
    'penalty': 'l2',
    'solver': 'saga'
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
joblib.dump(tfidf_vectorizer, f'{models_dir}/tfidf_ngram_pos_vectorizer.joblib')
joblib.dump(best_log_reg, f'{models_dir}/best_logistic_regression_ngram_pos.joblib')
print(f"Models saved to {models_dir}/")

# Save training parameters to a file
with open(f'{results_dir}/tfidf_ngram_pos_training_info.txt', 'w') as f:
    f.write("TF-IDF Vectorizer Parameters (With POS Tagging):\n")
    for param, value in tfidf_vectorizer.get_params().items():
        f.write(f"{param}: {value}\n")
    
    f.write("\nLogistic Regression Parameters:\n")
    for param, value in best_log_reg.get_params().items():
        f.write(f"{param}: {value}\n")
    
    f.write(f"\nTraining set size: {train_df.shape[0]} samples\n")
    f.write(f"Number of features: {X_train.shape[1]}\n")

print(f"\nTraining information saved to '{results_dir}/tfidf_ngram_pos_training_info.txt'")
print("\nTF-IDF with N-grams and POS Tagging Training Complete!") 