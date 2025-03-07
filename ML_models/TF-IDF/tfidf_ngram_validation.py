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

print(f"Validation set size: {val_df.shape[0]} ({val_df.shape[0]/df.shape[0]:.2%})")

# Create directories for saving models and results
models_dir = 'ML_models/saved_models'
results_dir = 'results'
os.makedirs(models_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)

# Check if we have a saved model, if not, train a new one with best parameters
try:
    print("\nAttempting to load trained TF-IDF vectorizer and model...")
    tfidf_vectorizer = joblib.load(f'{models_dir}/tfidf_ngram_vectorizer.joblib')
    best_log_reg = joblib.load(f'{models_dir}/best_logistic_regression_ngram.joblib')
    print("Models loaded successfully!")
except FileNotFoundError:
    print("Models not found. Training new model with best parameters...")
    
    # TF-IDF Vectorization with n-grams
    print("\nApplying TF-IDF vectorization with unigrams, bigrams, and trigrams...")
    tfidf_vectorizer = TfidfVectorizer(
        tokenizer=tokenize_text,
        max_features=10000,
        min_df=3,
        max_df=0.9,
        ngram_range=(1, 3),
        sublinear_tf=True
    )
    
    X_train = tfidf_vectorizer.fit_transform(train_df['text_'])
    
    # Use the best parameters found from previous grid search
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
    best_log_reg.fit(X_train, train_df['label'])
    
    # Save the model and vectorizer
    print("\nSaving trained model and vectorizer...")
    joblib.dump(tfidf_vectorizer, f'{models_dir}/tfidf_ngram_vectorizer.joblib')
    joblib.dump(best_log_reg, f'{models_dir}/best_logistic_regression_ngram.joblib')
    print(f"Models saved to {models_dir}/")

# Transform validation data
X_val = tfidf_vectorizer.transform(val_df['text_'])

# Evaluate on validation set
print("\nEvaluating model on validation set...")
val_predictions = best_log_reg.predict(X_val)
val_accuracy = accuracy_score(val_df['label'], val_predictions)
val_report = classification_report(val_df['label'], val_predictions)

print("\n🔹 Validation Set Evaluation:")
print(val_report)
print(f"Validation Accuracy: {val_accuracy:.4f}")

# Create confusion matrix for validation set
cm = confusion_matrix(val_df['label'], val_predictions)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=best_log_reg.classes_, 
            yticklabels=best_log_reg.classes_)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix (Validation Set)')
plt.tight_layout()
plt.savefig(f'{results_dir}/confusion_matrix_tfidf_ngrams_validation.png')
print(f"\nConfusion matrix saved as '{results_dir}/confusion_matrix_tfidf_ngrams_validation.png'")

# Get the most important features for each class
feature_names = tfidf_vectorizer.get_feature_names_out()
coef = best_log_reg.coef_[0]  # For binary classification

# Top features for each class
top_n = 20
top_positive_indices = np.argsort(coef)[-top_n:]
top_negative_indices = np.argsort(coef)[:top_n]

print(f"\n🔹 Top {top_n} features for class '{best_log_reg.classes_[1]}':")
for idx in reversed(top_positive_indices):
    print(f"{feature_names[idx]}: {coef[idx]:.4f}")

print(f"\n🔹 Top {top_n} features for class '{best_log_reg.classes_[0]}':")
for idx in top_negative_indices:
    print(f"{feature_names[idx]}: {coef[idx]:.4f}")

# Save validation results to a file
with open(f'{results_dir}/tfidf_ngram_validation_results.txt', 'w') as f:
    f.write(f"Validation Accuracy: {val_accuracy:.4f}\n\n")
    f.write("Classification Report:\n")
    f.write(val_report)
    f.write("\n\nModel Parameters:\n")
    for param, value in best_log_reg.get_params().items():
        f.write(f"{param}: {value}\n")
    
    f.write("\nTop features for class '{}':\n".format(best_log_reg.classes_[1]))
    for idx in reversed(top_positive_indices):
        f.write(f"{feature_names[idx]}: {coef[idx]:.4f}\n")
    
    f.write("\nTop features for class '{}':\n".format(best_log_reg.classes_[0]))
    for idx in top_negative_indices:
        f.write(f"{feature_names[idx]}: {coef[idx]:.4f}\n")

print(f"\nValidation results saved to '{results_dir}/tfidf_ngram_validation_results.txt'")
print("\nTF-IDF with N-grams Validation Complete!") 