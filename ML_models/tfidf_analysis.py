import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Load the dataset
file_path = r"C:\Users\selka\OneDrive\Bureaublad\Machine_Learning\repository\MachineLearningProject\datasets\fake_reviews_dataset.csv"  # Ensure the file is in the same directory
df = pd.read_csv(file_path, encoding="utf-8")

# Load English stopwords
stop_words = set(stopwords.words("english"))

# Function to clean and tokenize text
def process_text(text):
    tokens = word_tokenize(text.lower())  # Convert to lowercase and tokenize
    filtered_tokens = [word for word in tokens if word not in stop_words and word.isalpha()]  # Remove stopwords and non-alphabetic tokens
    return " ".join(filtered_tokens)  # Convert back to string for TF-IDF

# Apply text processing
df["clean_text"] = df["text_"].apply(process_text)

# Separate CG (AI-generated) and OR (human-written) reviews
cg_texts = df[df["label"] == "CG"]["clean_text"].tolist()
or_texts = df[df["label"] == "OR"]["clean_text"].tolist()

# Combine texts and create labels
texts = cg_texts + or_texts
labels = ["CG"] * len(cg_texts) + ["OR"] * len(or_texts)

# Compute TF-IDF
vectorizer = TfidfVectorizer(max_features=1000)
tfidf_matrix = vectorizer.fit_transform(texts)

# Extract feature names (words)
feature_names = vectorizer.get_feature_names_out()

# Compute average TF-IDF scores for CG and OR
cg_tfidf = tfidf_matrix[:len(cg_texts)].mean(axis=0)
or_tfidf = tfidf_matrix[len(cg_texts):].mean(axis=0)

# Get top words for each category
cg_top_indices = cg_tfidf.argsort().tolist()[0][-10:]  # Top 10 AI (CG) words
or_top_indices = or_tfidf.argsort().tolist()[0][-10:]  # Top 10 Human (OR) words

cg_top_words = [feature_names[i] for i in cg_top_indices]
or_top_words = [feature_names[i] for i in or_top_indices]

# Print results
print("🔹 **Top 10 Unique AI-Generated (CG) Words:**")
print(", ".join(cg_top_words))

print("\n🔹 **Top 10 Unique Human (OR) Words:**")
print(", ".join(or_top_words))
