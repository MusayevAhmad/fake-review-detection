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
    print("\nLoading trained TF-IDF vectorizer and best model...")
    tfidf_vectorizer = joblib.load(f'{models_dir}/tfidf_ngram_vectorizer.joblib')
    best_log_reg = joblib.load(f'{models_dir}/best_logistic_regression_ngram.joblib')
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
print("\n🔹 Test Set Evaluation:")
print(classification_report(test_df['label'], test_predictions))
print(f"Test Accuracy: {test_accuracy:.4f}")

# Create confusion matrix for test set
cm = confusion_matrix(test_df['label'], test_predictions)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=best_log_reg.classes_, 
            yticklabels=best_log_reg.classes_)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix (Test Set)')
plt.tight_layout()
plt.savefig(f'{results_dir}/confusion_matrix_tfidf_ngrams_test.png')
print(f"\nConfusion matrix saved as '{results_dir}/confusion_matrix_tfidf_ngrams_test.png'")

# Save test results to a file
with open(f'{results_dir}/tfidf_ngram_test_results.txt', 'w') as f:
    f.write(f"Test Accuracy: {test_accuracy:.4f}\n\n")
    f.write("Classification Report:\n")
    f.write(classification_report(test_df['label'], test_predictions))
    
print(f"\nTest results saved to '{results_dir}/tfidf_ngram_test_results.txt'")
print("\nTF-IDF with N-grams Test Evaluation Complete!")