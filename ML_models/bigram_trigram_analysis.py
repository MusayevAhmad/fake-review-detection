import pandas as pd
import nltk
from nltk.util import ngrams
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Make sure you've downloaded stopwords:
# nltk.download('stopwords')

# Load dataset
file_path = '/Users/ahmedmusayev/Desktop/VU/Year 2/P4/ML/MachineLearningProject/datasets/fake_reviews_dataset.csv'
df = pd.read_csv(file_path, encoding="utf-8")

# Define stopwords set
stop_words = set(stopwords.words('english'))

# Tokenization function that removes stopwords
def tokenize_text(text):
    tokens = word_tokenize(text.lower())  # Lowercase & tokenize
    # Keep only alphabetic tokens and remove stopwords
    tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
    return tokens

# Apply tokenization
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
print("\n🔹 **Top 10 AI-Generated (CG) Bigrams (Stopwords Removed):**")
for bigram, count in cg_bigrams:
    print(f"{bigram}: {count}")

print("\n🔹 **Top 10 AI-Generated (CG) Trigrams (Stopwords Removed):**")
for trigram, count in cg_trigrams:
    print(f"{trigram}: {count}")

print("\n🔹 **Top 10 Human (OR) Bigrams (Stopwords Removed):**")
for bigram, count in or_bigrams:
    print(f"{bigram}: {count}")

print("\n🔹 **Top 10 Human (OR) Trigrams (Stopwords Removed):**")
for trigram, count in or_trigrams:
    print(f"{trigram}: {count}")