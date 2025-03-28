# filepath: /Users/ahmedmusayev/Desktop/VU/Year 2/P4/ML/MachineLearningProject/DL_models/single_review_inference_gui.py

import tkinter as tk
from tkinter import messagebox, scrolledtext
import tkinter.font as font
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# 1. Load Model & Tokenizer
model_checkpoint_path = "bert-fake-vs-real/checkpoint-16176"  # Update if needed
model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint_path)
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint_path)

# Label map must match your training
label_map = {0: "OR Human Generated", 1: "CG Computer Generated"}

# -------------------------
# Functions
# -------------------------
def predict_review():
    review_text = text_box.get("1.0", tk.END).strip()
    if not review_text:
        messagebox.showwarning("Warning", "Please enter some text.")
        return
    
    encoded_input = tokenizer(review_text, truncation=True, padding="max_length", max_length=128, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**encoded_input)
        logits = outputs.logits
        predicted_label_id = torch.argmax(logits, dim=-1).item()
    
    predicted_label = label_map[predicted_label_id]
    messagebox.showinfo("Prediction Result", f"Predicted Label:\n{predicted_label}")

def clear_text():
    text_box.delete("1.0", tk.END)

# -------------------------
# Main Window Setup
# -------------------------
root = tk.Tk()
root.title("BERT Review Classifier")
root.geometry("600x400")
root.config(bg="#fafafa")

# Custom font
title_font = font.Font(family="Helvetica", size=14, weight="bold")
button_font = font.Font(family="Helvetica", size=10)

# Title Label
label_title = tk.Label(root, text="BERT Single Review Classifier", bg="#fafafa", font=title_font)
label_title.pack(pady=10)

label_prompt = tk.Label(root, text="Enter your review below:", bg="#fafafa")
label_prompt.pack()

# Scrolled Text Box
text_box = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=60, height=10)
text_box.pack(padx=10, pady=5)

# Button Frame
button_frame = tk.Frame(root, bg="#fafafa")
button_frame.pack(pady=10)

predict_button = tk.Button(button_frame, text="Predict", command=predict_review, bg="#4caf50", fg="black", font=button_font)
predict_button.grid(row=0, column=0, padx=10)

clear_button = tk.Button(button_frame, text="Clear", command=clear_text, bg="#f44336", fg="black", font=button_font)
clear_button.grid(row=0, column=1, padx=10)

# Start the GUI event loop
root.mainloop()
