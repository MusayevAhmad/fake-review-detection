import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Make sure you've downloaded necessary NLTK resources
# nltk.download('punkt')
# nltk.download('stopwords')

# Load dataset
file_path = '/Users/ahmedmusayev/Desktop/VU/Year 2/P4/ML/MachineLearningProject/datasets/fake_reviews_dataset.csv'
df = pd.read_csv(file_path, encoding="utf-8")

# Define stopwords set
stop_words = set(stopwords.words('english'))

# Tokenization function that removes stopwords
def tokenize_text(text):
    if isinstance(text, str):
        tokens = word_tokenize(text.lower())  # Lowercase & tokenize
        # Keep only alphabetic tokens and remove stopwords
        tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
        return tokens
    return []

# Print dataset information
print(f"Dataset shape: {df.shape}")
print(f"Label distribution:\n{df['label'].value_counts()}")

# Split the data into training, validation, and test sets (80/10/10)
train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['label'])

print(f"\nTraining set size: {train_df.shape[0]} ({train_df.shape[0]/df.shape[0]:.2%})")
print(f"Validation set size: {val_df.shape[0]} ({val_df.shape[0]/df.shape[0]:.2%})")
print(f"Test set size: {test_df.shape[0]} ({test_df.shape[0]/df.shape[0]:.2%})")

# TF-IDF Vectorization
print("\nApplying TF-IDF vectorization...")
tfidf_vectorizer = TfidfVectorizer(
    tokenizer=tokenize_text,
    max_features=5000,
    min_df=5,  # Minimum document frequency
    max_df=0.8  # Maximum document frequency (as a percentage)
)

X_train = tfidf_vectorizer.fit_transform(train_df['text_'])
X_val = tfidf_vectorizer.transform(val_df['text_'])
X_test = tfidf_vectorizer.transform(test_df['text_'])

print(f"TF-IDF features: {X_train.shape[1]}")

# Logistic Regression Model
print("\nTraining logistic regression model...")
log_reg = LogisticRegression(max_iter=1000, C=1.0, class_weight='balanced')
log_reg.fit(X_train, train_df['label'])

# Evaluate on validation set
val_predictions = log_reg.predict(X_val)
val_accuracy = accuracy_score(val_df['label'], val_predictions)
print("\n🔹 Validation Set Evaluation:")
print(classification_report(val_df['label'], val_predictions))
print(f"Validation Accuracy: {val_accuracy:.4f}")

# Evaluate on test set
test_predictions = log_reg.predict(X_test)
test_accuracy = accuracy_score(test_df['label'], test_predictions)
print("\n🔹 Test Set Evaluation:")
print(classification_report(test_df['label'], test_predictions))
print(f"Test Accuracy: {test_accuracy:.4f}")

# Create confusion matrix for test set
cm = confusion_matrix(test_df['label'], test_predictions)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=log_reg.classes_, 
            yticklabels=log_reg.classes_)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix (Test Set)')
plt.tight_layout()
plt.savefig('confusion_matrix_tfidf.png')
print("\nConfusion matrix saved as 'confusion_matrix_tfidf.png'")

# Get the most important features for each class
feature_names = tfidf_vectorizer.get_feature_names_out()
coef = log_reg.coef_[0]  # For binary classification

# Top features for each class
top_n = 20
top_positive_indices = np.argsort(coef)[-top_n:]
top_negative_indices = np.argsort(coef)[:top_n]

print(f"\n🔹 Top {top_n} features for class '{log_reg.classes_[1]}':")
for idx in reversed(top_positive_indices):
    print(f"{feature_names[idx]}: {coef[idx]:.4f}")

print(f"\n🔹 Top {top_n} features for class '{log_reg.classes_[0]}':")
for idx in top_negative_indices:
    print(f"{feature_names[idx]}: {coef[idx]:.4f}")

print("\nTF-IDF with Logistic Regression analysis complete!") 