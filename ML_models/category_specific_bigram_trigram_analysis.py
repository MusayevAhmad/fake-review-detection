import pandas as pd
import nltk
from nltk.util import ngrams
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Make sure you've downloaded the necessary NLTK data:
# nltk.download('punkt')
# nltk.download('stopwords')

# Load dataset
file_path = '/Users/ahmedmusayev/Desktop/VU/Year 2/P4/ML/MachineLearningProject/datasets/fake_reviews_dataset.csv'
df = pd.read_csv(file_path, encoding="utf-8")

# Define stopwords set
stop_words = set(stopwords.words("english"))

# Tokenization function that removes stopwords
def tokenize_text(text):
    tokens = word_tokenize(text.lower())  # Lowercase & tokenize
    tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
    return tokens

# Apply tokenization
df["tokens"] = df["text_"].apply(tokenize_text)

# Function to extract the most common n-grams
def get_common_ngrams(text_list, n=2, top_n=10):
    """
    text_list: List of lists of tokens, e.g. df["tokens"].tolist()
    n: the n-gram size (2 for bigrams, 3 for trigrams, etc.)
    top_n: how many of the most common n-grams to return
    """
    all_ngrams = [ngram for tokens in text_list for ngram in ngrams(tokens, n)]
    return Counter(all_ngrams).most_common(top_n)

# Get all unique categories
unique_categories = df["category"].unique()

# For each category, compute and print top bigrams & trigrams
for cat in unique_categories:
    cat_subset = df[df["category"] == cat]
    token_lists = cat_subset["tokens"].tolist()

    # Get top 10 bigrams & trigrams
    cat_bigrams = get_common_ngrams(token_lists, n=2, top_n=10)
    cat_trigrams = get_common_ngrams(token_lists, n=3, top_n=10)

    # Print results for this category
    print(f"\n=== Category: {cat} ===")
    print("Top 10 Bigrams (Stopwords Removed):")
    for bigram, count in cat_bigrams:
        print(f"  {bigram}: {count}")

    print("\nTop 10 Trigrams (Stopwords Removed):")
    for trigram, count in cat_trigrams:
        print(f"  {trigram}: {count}")