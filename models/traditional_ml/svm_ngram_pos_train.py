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
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
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

# Set up SVM model with predefined parameters
# You can modify these based on grid search results or your preferences
print("\nSetting up SVM model with predefined parameters...")
kernel_type = 'linear'  # 'linear' or 'rbf'
c_value = 10.0         # 0.1, 1.0, or 10.0
gamma_value = 'scale'  # 'scale' or 0.1

svm_params = {
    'C': 10.0,
    'kernel': 'rbf',
    'gamma': 'scale',
    'class_weight': 'balanced',
    'probability': True,
    'random_state': 42
}

print(f"Using parameters: {svm_params}")

# Train SVM model
print("\nTraining SVM model on training data...")
svm_model = SVC(**svm_params)
svm_model.fit(X_train, train_df['label'])
print("SVM model training complete!")

# Evaluate on validation set
print("\nEvaluating model on validation set...")
X_val = tfidf_vectorizer.transform(val_df['text_'])
val_predictions = svm_model.predict(X_val)
val_accuracy = accuracy_score(val_df['label'], val_predictions)
val_report = classification_report(val_df['label'], val_predictions)

print("\n🔹 Validation Set Evaluation (SVM with POS Tagging):")
print(val_report)
print(f"Validation Accuracy: {val_accuracy:.8f}")

# Create confusion matrix for validation set
cm = confusion_matrix(val_df['label'], val_predictions)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=svm_model.classes_, 
            yticklabels=svm_model.classes_)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title(f'Confusion Matrix (Validation Set - SVM {kernel_type} C={c_value})')
plt.tight_layout()

# Create directories for saving models and results
models_dir = 'ML_models/saved_models'
results_dir = 'results'
os.makedirs(models_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)

# Save confusion matrix
plt.savefig(f'{results_dir}/confusion_matrix_svm_{kernel_type}_C{c_value}_validation.png')
print(f"Confusion matrix saved to '{results_dir}/confusion_matrix_svm_{kernel_type}_C{c_value}_validation.png'")

# Save the model and vectorizer
print("\nSaving trained model and vectorizer...")
joblib.dump(tfidf_vectorizer, f'{models_dir}/svm_ngram_pos_vectorizer.joblib')
model_filename = f'{models_dir}/svm_ngram_pos_{kernel_type}_C{c_value}.joblib'
joblib.dump(svm_model, model_filename)
print(f"Model saved as: {model_filename}")
print(f"Vectorizer saved as: {models_dir}/svm_ngram_pos_vectorizer.joblib")

# Calculate feature importance if using linear kernel
if kernel_type == 'linear':
    print("\nCalculating feature importance...")
    feature_names = tfidf_vectorizer.get_feature_names_out()
    coef = svm_model.coef_[0]
    
    # Convert sparse matrix to dense array if needed
    if hasattr(coef, "toarray"):
        coef = coef.toarray().flatten()
    
    top_n = 20
    top_positive_indices = np.argsort(coef)[-top_n:]
    top_negative_indices = np.argsort(coef)[:top_n]
    
    print(f"\n🔹 Top {top_n} features for class '{svm_model.classes_[1]}':")
    for idx in reversed(top_positive_indices):
        print(f"{feature_names[idx]}: {coef[idx]:.4f}")
    
    print(f"\n🔹 Top {top_n} features for class '{svm_model.classes_[0]}':")
    for idx in top_negative_indices:
        print(f"{feature_names[idx]}: {coef[idx]:.4f}")

# Save training results and parameters to a file
with open(f'{results_dir}/svm_ngram_pos_training_results_{kernel_type}_C{c_value}.txt', 'w') as f:
    f.write(f"SVM Training Results with POS Tagging (kernel={kernel_type}, C={c_value})\n")
    f.write("=====================================================\n\n")
    
    f.write("SVM Parameters:\n")
    for param, value in svm_params.items():
        f.write(f"{param}: {value}\n")
    
    f.write("\nValidation Results:\n")
    f.write(f"Validation Accuracy: {val_accuracy:.4f}\n\n")
    f.write("Classification Report:\n")
    f.write(val_report)
    
    f.write("\n\nTF-IDF Vectorizer Parameters:\n")
    for param, value in tfidf_vectorizer.get_params().items():
        f.write(f"{param}: {value}\n")
    
    f.write(f"\nTraining set size: {train_df.shape[0]} samples\n")
    f.write(f"Validation set size: {val_df.shape[0]} samples\n")
    f.write(f"Number of features: {X_train.shape[1]}\n")
    
    # Add timing information
    total_time = time.time() - start_time
    minutes = total_time / 60
    f.write(f"\nTotal execution time: {minutes:.1f} minutes\n")
    
    # Add feature importance if using linear kernel
    if kernel_type == 'linear':
        f.write("\nFeature Importance:\n")
        f.write(f"Top features for class '{svm_model.classes_[1]}':\n")
        for idx in reversed(top_positive_indices):
            f.write(f"{feature_names[idx]}: {coef[idx]:.4f}\n")
        
        f.write(f"\nTop features for class '{svm_model.classes_[0]}':\n")
        for idx in top_negative_indices:
            f.write(f"{feature_names[idx]}: {coef[idx]:.4f}\n")

print(f"\nTraining results saved to '{results_dir}/svm_ngram_pos_training_results_{kernel_type}_C{c_value}.txt'")
print("\nSVM with TF-IDF N-grams and POS Tagging Training Complete!")
print(f"\nTotal execution time: {(time.time() - start_time) / 60:.1f} minutes")