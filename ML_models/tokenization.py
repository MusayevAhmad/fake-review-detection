import csv
import nltk

# Ensure the necessary tokenizer is downloaded
nltk.download('punkt')
nltk.download('punkt_tab')  # Try downloading the missing resource


from nltk.tokenize import word_tokenize

# Define the file path (update if necessary)
file_path = '/Users/ahmedmusayev/Desktop/VU/Year 2/P4/ML/MachineLearningProject/datasets/fake_reviews_dataset.csv'
# Read and tokenize the CSV file
with open(file_path, newline='', encoding='utf-8') as f:
    reader = csv.reader(f)
    headers = next(reader)  # Read header row
    print(f"Headers: {headers}\n")

    for row in reader:
        tokenized_row = [word_tokenize(cell) for cell in row]
        print(f"Original: {row}")
        print(f"Tokenized: {tokenized_row}\n")
