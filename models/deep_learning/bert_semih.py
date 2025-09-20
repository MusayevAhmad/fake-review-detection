"""
BERT Transformer Model for Fake Review Detection

This script fine-tunes a pre-trained BERT model for binary classification:
- Uses BERT-base-uncased as the foundation model
- Fine-tunes with custom classification head
- Implements proper train/validation/test splits
- Saves checkpoints for inference

Author: Group 86  
Date: 2024-2025
Course: VU Machine Learning
"""

import pandas as pd
import numpy as np
import torch
import os

# Hugging Face libraries
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments
from datasets import Dataset, DatasetDict

# For splitting, metrics, etc.
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

###############################################################################
# 1. LOAD YOUR CSV
###############################################################################
csv_path = 'data/fake_reviews_dataset.csv'
df = pd.read_csv(csv_path, encoding="utf-8")

# Drop any rows where text is missing
df.dropna(subset=['text_'], inplace=True)

# Map labels: CG -> 1, OR -> 0 (binary classification)
df['label_num'] = df['label'].map({"CG": 1, "OR": 0})

###############################################################################
# 2. SPLIT: 80% TRAIN, 10% VAL, 10% TEST
###############################################################################
# First split: separate out 80% train from 20% temp
train_df, temp_df = train_test_split(
    df, 
    test_size=0.2, 
    random_state=42, 
    stratify=df['label_num']
)

# Second split: from that 20% "temp", we do 50-50 for val & test
val_df, test_df = train_test_split(
    temp_df, 
    test_size=0.5, 
    random_state=42, 
    stratify=temp_df['label_num']
)

print(f"Train size: {len(train_df)}")
print(f"Val size:   {len(val_df)}")
print(f"Test size:  {len(test_df)}")

###############################################################################
# 3. CREATE HUGGING FACE DATASETS
###############################################################################
train_dataset = Dataset.from_pandas(train_df, preserve_index=False)
val_dataset   = Dataset.from_pandas(val_df, preserve_index=False)
test_dataset  = Dataset.from_pandas(test_df, preserve_index=False)

# Combine into a single DatasetDict for convenience (optional)
dataset = DatasetDict({
    'train': train_dataset,
    'validation': val_dataset,
    'test': test_dataset
})

###############################################################################
# 4. TOKENIZATION
###############################################################################
model_name = "distilbert-base-uncased"  # or "bert-base-uncased", etc.
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_function(example):
    return tokenizer(
        example["text_"],
        truncation=True,
        padding="max_length",
        max_length=128 
    )

# Apply tokenization to each subset in the dataset
encoded_dataset = dataset.map(tokenize_function, batched=True)

###############################################################################
# 5. PREP FOR TRAINING
###############################################################################
# Rename "label_num" to "labels" so HF Trainer recognizes it as the target
encoded_dataset = encoded_dataset.rename_column("label_num", "labels")

# Remove any columns we don't need for training
# If your CSV has "text_", "label", etc., we can remove them from the input
encoded_dataset = encoded_dataset.remove_columns(["label", "text_"])

# Convert to torch Tensors
encoded_dataset.set_format("torch")

###############################################################################
# 6. INIT MODEL
###############################################################################
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

###############################################################################
# 7. TRAINING ARGUMENTS
###############################################################################
training_args = TrainingArguments(
    output_dir="bert-fake-vs-real",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir="bert-logs",
    num_train_epochs=4,        # Increase epochs for more training
    per_device_train_batch_size=8,  # Smaller batch size can improve generalization
    per_device_eval_batch_size=8,
    learning_rate=5e-5,        # Typical range: 2e-5 to 5e-5
    weight_decay=0.01,         # Regularization to reduce overfitting
    load_best_model_at_end=True,
    metric_for_best_model="accuracy"
)

###############################################################################
# 8. METRICS FOR VALIDATION
###############################################################################
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall
    }

###############################################################################
# 9. CREATE TRAINER
###############################################################################
from transformers import Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_dataset["train"],       # 80% training
    eval_dataset=encoded_dataset["validation"],   # 10% validation
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

###############################################################################
# 10. TRAIN ON 80%, VALIDATE ON 10%
###############################################################################
trainer.train()  # This will automatically evaluate on the validation set each epoch

###############################################################################
# 11. CHECK VALIDATION PERFORMANCE
###############################################################################
# Evaluate the final model on the 10% validation set
val_metrics = trainer.evaluate(encoded_dataset["validation"])
print("\nValidation Results:")
print(val_metrics)

# At this point, you can:
# - Adjust hyperparameters (e.g., num_train_epochs, batch_size)
# - Re-train until you're happy with the validation performance
