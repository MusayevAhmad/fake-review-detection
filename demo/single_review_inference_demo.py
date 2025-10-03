# Single Review Inference Demo

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# 1. Load your saved model checkpoint + tokenizer
model_checkpoint_path = "models/saved_models/bert_checkpoints/checkpoint-16176"
model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint_path)
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint_path)

# 2. Prepare your input text (a single review)
text = "I recently stayed at The Talbott Hotel for 3 nights and I could not have been more disappointed. It was a terrible experience. Long wait to check into room since room wasn't ready even though I arrived way after the check in time. Front desk was very rude. Was given a smoking room when I had requested non-smoking. Room service left a lot to be desired. The food took forever to arrive and when it did it was cold and unappetizing. The staff at the hotel was not very helpful and was very unfriendly. Had no toilet paper in room. Room was very noisy. Could hear people next door opening doors, watching tv, talking. The room was very dingy, small and had a musty odor. Needless to say I will not be staying here again nor will I be recommending this hotel to anyone else. Stay away from this hotel. Not worth the money."

# 3. Encode (tokenize) the single text
encoded_input = tokenizer(
    text,
    truncation=True,
    padding="max_length",
    max_length=128,
    return_tensors="pt"  # Return PyTorch tensors
)

# 4. Run the text through the model
with torch.no_grad():
    outputs = model(**encoded_input)
    logits = outputs.logits
    # Get the class with the highest score
    predicted_label_id = torch.argmax(logits, dim=-1).item()

# 5. Map back to label names
#   => Make sure it matches how you encoded label_num in training (OR=0, CG=1)
label_map = {0: "OR Human Generated", 1: "CG Computer Generated"}
predicted_label = label_map[predicted_label_id]

print(f"Review: {text}")
print(f"Predicted Label: {predicted_label}")
