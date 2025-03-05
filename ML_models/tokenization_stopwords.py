import csv
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Ensure necessary NLTK resources are available
nltk.download('punkt')
nltk.download('stopwords')

# Define file path
file_path = '/Users/ahmedmusayev/Desktop/VU/Year 2/P4/ML/MachineLearningProject/datasets/fake reviews dataset.csv'  # Update the file name if needed

# Load English stopwords
stop_words = set(stopwords.words('english'))

# Read and tokenize the CSV file
with open(file_path, newline='', encoding='utf-8') as f:
    reader = csv.reader(f)
    headers = next(reader)  # Read header row
    print(f"Headers: {headers}\n")

    for row in reader:
        tokenized_row = [word_tokenize(cell) for cell in row]  # Tokenize each column
        filtered_row = [[word for word in tokens if word.lower() not in stop_words] for tokens in tokenized_row]  # Remove stopwords

        print(f"Original: {row}")
        print(f"Tokenized: {tokenized_row}")
        print(f"Filtered (No Stopwords): {filtered_row}\n")
