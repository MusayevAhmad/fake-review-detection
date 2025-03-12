import pandas as pd
import nltk
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import time
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.util import ngrams
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Start timing
start_time = time.time()

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

# Function to extract n-grams
def get_common_ngrams(text_list, n=2, top_n=10):
    all_ngrams = []
    for tokens in text_list:
        n_grams = list(ngrams(tokens, n))
        all_ngrams.extend(n_grams)
    
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
    max_features=30000,  # Reduced from 50000 to speed up
    min_df=3,
    max_df=0.9,
    ngram_range=(1, 3),
    sublinear_tf=False
)

X_train = tfidf_vectorizer.fit_transform(train_df['text_'])
print(f"TF-IDF features: {X_train.shape[1]}")

# Define an efficient but comprehensive parameter grid
print("\nSetting up grid search with optimized parameter grid...")
param_grid = {
    # Focus on key parameters while keeping the grid small
    'C': [0.1, 1.0, 10.0],      # 3 values instead of 4
    'kernel': ['linear', 'rbf'], # Both key kernels
    'gamma': ['scale', 0.1],     # 2 values instead of 3 
    'class_weight': ['balanced'] # Only balanced to handle class imbalance
}

# First try with a sample to estimate full run time
print("\nUsing a small subset to estimate full grid search time...")
X_train_sample = X_train[:min(1000, X_train.shape[0])]
y_train_sample = train_df['label'][:min(1000, X_train.shape[0])]

grid_sample = GridSearchCV(
    SVC(probability=True, random_state=42),
    param_grid,
    cv=2,           # Smaller CV for timing
    verbose=0,
    n_jobs=1        # Single job for timing
)

sample_start = time.time()
grid_sample.fit(X_train_sample, y_train_sample)
sample_duration = time.time() - sample_start

# Estimate full run time
estimated_multiplier = X_train.shape[0] / X_train_sample.shape[0]
estimated_full_time = sample_duration * estimated_multiplier * 3  # cv=3 / cv=2 * parallelization factor

hours = estimated_full_time // 3600
minutes = (estimated_full_time % 3600) // 60
print(f"\nEstimated full grid search time: {hours:.0f} hours and {minutes:.0f} minutes")
print("This is a rough estimate and could vary based on system load")

# Let user know this will take time
print("\nPerforming full grid search. This might take several hours...")
print("Please do not shut down your computer; the process will continue running.")

# Create GridSearchCV object
grid_search = GridSearchCV(
    SVC(probability=True, random_state=42),
    param_grid,
    cv=3,                # 3-fold CV for speed (instead of 5)
    n_jobs=-1,           # Use all CPU cores
    verbose=2,           # Detailed output
    scoring='accuracy',  # Primary metric
    return_train_score=True
)

# Fit grid search
grid_search.fit(X_train, train_df['label'])

# Get best parameters and model
best_params = grid_search.best_params_
best_score = grid_search.best_score_
best_svm = grid_search.best_estimator_

print(f"\nBest parameters: {best_params}")
print(f"Best cross-validation score: {best_score:.4f}")

# Create directories for saving models and results
models_dir = 'ML_models/saved_models'
results_dir = 'results'
os.makedirs(models_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)

# Save the best model and vectorizer
print("\nSaving trained model and vectorizer...")
joblib.dump(tfidf_vectorizer, f'{models_dir}/svm_ngram_pos_vectorizer.joblib')
model_filename = f'{models_dir}/svm_ngram_pos_{best_params["kernel"]}_C{best_params["C"]}_best.joblib'
joblib.dump(best_svm, model_filename)
print(f"Best model saved as: {model_filename}")
print(f"Vectorizer saved as: {models_dir}/svm_ngram_pos_vectorizer.joblib")

# Save grid search results and parameters to a file
with open(f'{results_dir}/svm_ngram_pos_grid_search_results.txt', 'w') as f:
    f.write("SVM Grid Search Results with POS Tagging\n")
    f.write("=======================================\n\n")
    
    f.write("Best Parameters:\n")
    for param, value in best_params.items():
        f.write(f"{param}: {value}\n")
    
    f.write(f"\nBest cross-validation score: {best_score:.4f}\n")
    
    f.write("\nAll Tested Parameters and Scores:\n")
    f.write("--------------------------------\n")
    
    # Get all grid search results
    results = grid_search.cv_results_
    for i in range(len(results['params'])):
        params = results['params'][i]
        mean_test = results['mean_test_score'][i]
        mean_train = results['mean_train_score'][i]
        
        f.write(f"\nParameters: {params}\n")
        f.write(f"Mean validation score: {mean_test:.4f}\n")
        f.write(f"Mean training score: {mean_train:.4f}\n")
        f.write(f"Overfitting gap: {mean_train - mean_test:.4f}\n")
    
    f.write("\n\nTF-IDF Vectorizer Parameters:\n")
    for param, value in tfidf_vectorizer.get_params().items():
        f.write(f"{param}: {value}\n")
    
    f.write(f"\nTraining set size: {train_df.shape[0]} samples\n")
    f.write(f"Number of features: {X_train.shape[1]}\n")
    
    # Add timing information
    total_time = time.time() - start_time
    hours = total_time // 3600
    minutes = (total_time % 3600) // 60
    f.write(f"\nTotal execution time: {hours:.0f} hours, {minutes:.0f} minutes\n")

print(f"\nGrid search results saved to '{results_dir}/svm_ngram_pos_grid_search_results.txt'")
print("\nSVM with TF-IDF N-grams and POS Tagging Grid Search Complete!")
print(f"\nTotal execution time: {(time.time() - start_time) / 60:.1f} minutes")