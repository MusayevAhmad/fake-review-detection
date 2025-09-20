import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import joblib
import os

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

# Split the data into training, validation, and test sets (80/10/10)
print("\nSplitting data into training, validation, and test sets...")
train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['label'])

print(f"Test set size: {test_df.shape[0]} ({test_df.shape[0]/df.shape[0]:.2%})")

# Load the trained vectorizer and model
models_dir = 'ML_models/saved_models'
os.makedirs(models_dir, exist_ok=True)

# Create results directory if it doesn't exist
results_dir = 'results'
os.makedirs(results_dir, exist_ok=True)

try:
    print("\nLoading trained TF-IDF vectorizer and best model (with POS tagging)...")
    tfidf_vectorizer = joblib.load(f'{models_dir}/tfidf_ngram_pos_vectorizer.joblib')
    best_log_reg = joblib.load(f'{models_dir}/best_logistic_regression_ngram_pos.joblib')
    print("Models loaded successfully!")
except FileNotFoundError:
    print("Error: Trained models not found. Please run the training script first.")
    exit(1)

# Transform test data
X_test = tfidf_vectorizer.transform(test_df['text_'])

# Evaluate on test set
print("\nEvaluating model on test set...")
test_predictions = best_log_reg.predict(X_test)
test_accuracy = accuracy_score(test_df['label'], test_predictions)
test_report = classification_report(test_df['label'], test_predictions)

print("\n🔹 Test Set Evaluation (With POS Tagging):")
print(test_report)
print(f"Test Accuracy: {test_accuracy:.4f}")

# Create confusion matrix for test set
cm = confusion_matrix(test_df['label'], test_predictions)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=best_log_reg.classes_, 
            yticklabels=best_log_reg.classes_)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix (Test Set - With POS Tagging)')
plt.tight_layout()
plt.savefig(f'{results_dir}/confusion_matrix_tfidf_ngrams_pos_test.png')
print(f"\nConfusion matrix saved as '{results_dir}/confusion_matrix_tfidf_ngrams_pos_test.png'")

# Save test results to a file
with open(f'{results_dir}/tfidf_ngram_pos_test_results.txt', 'w') as f:
    f.write(f"Test Accuracy (With POS Tagging): {test_accuracy:.4f}\n\n")
    f.write("Classification Report:\n")
    f.write(test_report)
    
print(f"\nTest results saved to '{results_dir}/tfidf_ngram_pos_test_results.txt'")
print("\nTF-IDF with N-grams and POS Tagging Test Evaluation Complete!")