import pandas as pd
import nltk
import json
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Download required NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Load the dataset
file_path = r"C:\Users\selka\OneDrive\Bureaublad\Machine_Learning\repository\MachineLearningProject\datasets\fake_reviews_dataset.csv"  # Make sure the file is in the same directory
df = pd.read_csv(file_path, encoding="utf-8")

# Load English stopwords
stop_words = set(stopwords.words("english"))

# Function to tokenize and remove stopwords
def process_text(text):
    tokens = word_tokenize(text)  # Tokenization
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words]  # Stopword removal
    return filtered_tokens

# Apply processing to CG and OR reviews
df["processed_text"] = df["text_"].apply(process_text)

# Split data into CG and OR categories
cg_reviews = df[df["label"] == "CG"][["processed_text"]].to_dict(orient="records")
or_reviews = df[df["label"] == "OR"][["processed_text"]].to_dict(orient="records")

# Create final JSON structure
final_data = {
    "CG": cg_reviews,
    "OR": or_reviews
}

# Save to JSON file
json_file_path = "processed_reviews.json"
with open(json_file_path, "w", encoding="utf-8") as json_file:
    json.dump(final_data, json_file, indent=4, ensure_ascii=False)

print(f"✅ Processed data saved to {json_file_path}")
