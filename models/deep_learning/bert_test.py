# filepath: /Users/ahmedmusayev/Desktop/VU/Year 2/P4/ML/MachineLearningProject/DL_models/bert_test.py

import pandas as pd
import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer

###############################################################################
# 1. LOAD TRAINED MODEL & TOKENIZER
###############################################################################
# Point to your best checkpoint (or final directory)
model_checkpoint_path = "bert-fake-vs-real/checkpoint-16176"
model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint_path)
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint_path)

###############################################################################
# 2. LOAD AND PREPARE TEST DATA
###############################################################################
csv_path = '/Users/ahmedmusayev/Desktop/VU/Year 2/P4/ML/MachineLearningProject/datasets/fake_reviews_dataset.csv'
df = pd.read_csv(csv_path, encoding="utf-8")

# Drop any rows where text is missing
df.dropna(subset=['text_'], inplace=True)

# Map labels: CG -> 1, OR -> 0 (binary classification)
df['label_num'] = df['label'].map({"CG": 1, "OR": 0})

# Split out just the 10% test portion (replicating your training splits)
# Adjust these if your test split is stored elsewhere.
# For demonstration, using the same approach:
train_temp, test_df = np.split(
    df.sample(frac=1, random_state=42), 
    [int(.8*len(df))]  # first 80%, then 20% test
)
# Or if you want to replicate exactly the 80/10/10 approach, load from disk or re-split

# Create Hugging Face dataset for test
test_dataset = Dataset.from_pandas(test_df, preserve_index=False)
encoded_test = DatasetDict({"test": test_dataset})

###############################################################################
# 3. TOKENIZE AND FORMAT TEST DATA
###############################################################################
def tokenize_function(example):
    return tokenizer(
        example["text_"],
        truncation=True,
        padding="max_length",
        max_length=128
    )

encoded_test = encoded_test.map(tokenize_function, batched=True)

# Rename "label_num" to "labels" (if it’s still present)
if "label_num" in encoded_test["test"].column_names:
    encoded_test = encoded_test.rename_column("label_num", "labels")

# Remove unneeded columns
columns_to_remove = [col for col in encoded_test["test"].column_names if col not in ["labels", "input_ids", "attention_mask"]]
encoded_test = encoded_test.remove_columns(columns_to_remove)
encoded_test.set_format("torch")

###############################################################################
# 4. CREATE A TEST Trainer
###############################################################################
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
    acc = accuracy_score(labels, preds)
    return {
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall
    }

trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

###############################################################################
# 5. RUN PREDICTIONS AND PRINT METRICS
###############################################################################
test_results = trainer.predict(encoded_test["test"])
print("\nTest Results:")
print(test_results.metrics)
