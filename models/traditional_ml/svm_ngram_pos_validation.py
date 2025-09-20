import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.svm import SVC
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

print(f"Validation set size: {val_df.shape[0]} ({val_df.shape[0]/df.shape[0]:.2%})")

# Load the trained vectorizer and model
models_dir = 'ML_models/saved_models'
results_dir = 'results'
os.makedirs(models_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)

# Define the model filename based on parameters you're evaluating
kernel_type = 'rbf'  # Change this based on which model you want to evaluate
c_value = 1
model_filename = f'{models_dir}/svm_ngram_pos_{kernel_type}_C{c_value}.joblib'

try:
    print(f"\nLoading trained TF-IDF vectorizer and SVM model with kernel={kernel_type}, C={c_value} (with POS tagging)...")
    tfidf_vectorizer = joblib.load(f'{models_dir}/svm_ngram_pos_vectorizer.joblib')
    best_svm = joblib.load(model_filename)
    print("Models loaded successfully!")
except FileNotFoundError:
    print(f"Error: Trained model {model_filename} not found. Please run the training script first.")
    exit(1)

# Transform validation data
X_val = tfidf_vectorizer.transform(val_df['text_'])

# Evaluate on validation set
print("\nEvaluating model on validation set...")
val_predictions = best_svm.predict(X_val)
val_accuracy = accuracy_score(val_df['label'], val_predictions)
val_report = classification_report(val_df['label'], val_predictions)

print("\n🔹 Validation Set Evaluation (SVM with POS Tagging):")
print(val_report)
print(f"Validation Accuracy: {val_accuracy:.4f}")

# Create confusion matrix for validation set
cm = confusion_matrix(val_df['label'], val_predictions)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=best_svm.classes_, 
            yticklabels=best_svm.classes_)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix (Validation Set - SVM with POS Tagging)')
plt.tight_layout()
plt.savefig(f'{results_dir}/confusion_matrix_svm_ngrams_pos_validation.png')
print(f"\nConfusion matrix saved as '{results_dir}/confusion_matrix_svm_ngrams_pos_validation.png'")

# Calculate feature importance (if using linear kernel)
if best_svm.kernel == 'linear' and hasattr(best_svm, 'coef_'):
    # Get feature importance from SVM coefficients
    feature_names = tfidf_vectorizer.get_feature_names_out()
    coef = best_svm.coef_[0]
    
    # Convert sparse matrix to dense array if needed
    if hasattr(coef, "toarray"):
        coef = coef.toarray().flatten()
    
    top_n = 20
    top_positive_indices = np.argsort(coef)[-top_n:]
    top_negative_indices = np.argsort(coef)[:top_n]
    
    print(f"\n🔹 Top {top_n} features for class '{best_svm.classes_[1]}':")
    for idx in reversed(top_positive_indices):
        print(f"{feature_names[idx]}: {coef[idx]:.4f}")
    
    print(f"\n🔹 Top {top_n} features for class '{best_svm.classes_[0]}':")
    for idx in top_negative_indices:
        print(f"{feature_names[idx]}: {coef[idx]:.4f}")
    
    # Save validation results to a file with feature importance
    with open(f'{results_dir}/svm_ngram_pos_validation_results.txt', 'w') as f:
        f.write(f"Validation Accuracy (SVM with POS Tagging): {val_accuracy:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(val_report)
        
        f.write("\n\nTop features for class '{}':\n".format(best_svm.classes_[1]))
        for idx in reversed(top_positive_indices):
            f.write(f"{feature_names[idx]}: {coef[idx]:.4f}\n")
        
        f.write("\nTop features for class '{}':\n".format(best_svm.classes_[0]))
        for idx in top_negative_indices:
            f.write(f"{feature_names[idx]}: {coef[idx]:.4f}\n")
else:
    # Save validation results to a file without feature importance
    with open(f'{results_dir}/svm_ngram_pos_validation_results.txt', 'w') as f:
        f.write(f"Validation Accuracy (SVM with POS Tagging): {val_accuracy:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(val_report)
        
        f.write("\nNote: Feature importance not available for non-linear kernel.")

print(f"\nValidation results saved to '{results_dir}/svm_ngram_pos_validation_results.txt'")
print("\nSVM with TF-IDF N-grams and POS Tagging Validation Evaluation Complete!")