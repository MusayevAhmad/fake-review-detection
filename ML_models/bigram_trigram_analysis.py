import pandas as pd
import nltk
from nltk.util import ngrams
from collections import Counter
from nltk.tokenize import word_tokenize

# Load dataset
file_path = r"C:\Users\selka\OneDrive\Bureaublad\Machine_Learning\repository\MachineLearningProject\datasets\fake_reviews_dataset.csv"  # Ensure the file is in the same directory
df = pd.read_csv(file_path, encoding="utf-8")

# Tokenization function
def tokenize_text(text):
    tokens = word_tokenize(text.lower())  # Lowercase & tokenize
    tokens = [word for word in tokens if word.isalpha()]  # Keep only words
    return tokens

# Apply tokenization
df["tokens"] = df["text_"].apply(tokenize_text)

# Function to extract bigrams & trigrams
def get_common_ngrams(text_list, n=2, top_n=10):
    all_ngrams = [ngram for tokens in text_list for ngram in ngrams(tokens, n)]
    return Counter(all_ngrams).most_common(top_n)

# Get bigrams & trigrams for CG and OR
cg_bigrams = get_common_ngrams(df[df["label"] == "CG"]["tokens"].tolist(), n=2)
cg_trigrams = get_common_ngrams(df[df["label"] == "CG"]["tokens"].tolist(), n=3)

or_bigrams = get_common_ngrams(df[df["label"] == "OR"]["tokens"].tolist(), n=2)
or_trigrams = get_common_ngrams(df[df["label"] == "OR"]["tokens"].tolist(), n=3)

# Print results
print("\n🔹 **Top 10 AI-Generated (CG) Bigrams:**")
for bigram, count in cg_bigrams:
    print(f"{bigram}: {count}")

print("\n🔹 **Top 10 AI-Generated (CG) Trigrams:**")
for trigram, count in cg_trigrams:
    print(f"{trigram}: {count}")

print("\n🔹 **Top 10 Human (OR) Bigrams:**")
for bigram, count in or_bigrams:
    print(f"{bigram}: {count}")

print("\n🔹 **Top 10 Human (OR) Trigrams:**")
for trigram, count in or_trigrams:
    print(f"{trigram}: {count}")
