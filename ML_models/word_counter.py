import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter

# Download required NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Load the dataset
file_path = r"C:\Users\selka\OneDrive\Bureaublad\Machine_Learning\repository\MachineLearningProject\datasets\fake_reviews_dataset.csv"  # Ensure the file is in the same directory
df = pd.read_csv(file_path, encoding="utf-8")

# Load English stopwords
stop_words = set(stopwords.words("english"))

# Function to tokenize and remove stopwords
def process_text(text):
    tokens = word_tokenize(text.lower())  # Convert to lowercase and tokenize
    filtered_tokens = [word for word in tokens if word not in stop_words and word.isalpha()]  # Remove stopwords and non-alphabetic tokens
    return filtered_tokens

# Apply text processing
df["processed_text"] = df["text_"].apply(process_text)

# Separate CG (AI-generated) and OR (human) reviews
cg_words = [word for tokens in df[df["label"] == "CG"]["processed_text"] for word in tokens]
or_words = [word for tokens in df[df["label"] == "OR"]["processed_text"] for word in tokens]

# Count the most common words
cg_top_words = Counter(cg_words).most_common(10)
or_top_words = Counter(or_words).most_common(10)

# Display results
print("🔹 **Top 10 AI-Generated (CG) Words:**")
for word, count in cg_top_words:
    print(f"{word}: {count}")

print("\n🔹 **Top 10 Human (OR) Words:**")
for word, count in or_top_words:
    print(f"{word}: {count}")
