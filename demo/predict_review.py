#!/usr/bin/env python3
"""
Simple command-line demo for fake review detection.

Usage:
    python demo/predict_review.py "This product is amazing! I love it so much."
    python demo/predict_review.py --model bert "Great quality and fast shipping!"
    python demo/predict_review.py --model tfidf "Absolutely fantastic product!"
"""

import argparse
import sys
import os

# Add the project root to the path to import modules
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

def predict_with_tfidf(text):
    """Predict using the TF-IDF + Logistic Regression model."""
    try:
        import joblib
        import pandas as pd
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        # Load saved model and vectorizer
        model_path = os.path.join(project_root, 'models', 'saved_models', 'best_logistic_regression_ngram.joblib')
        vectorizer_path = os.path.join(project_root, 'models', 'saved_models', 'tfidf_ngram_vectorizer.joblib')
        
        if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
            return "❌ TF-IDF model files not found. Please train the model first."
        
        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
        
        # Transform text and predict
        text_vector = vectorizer.transform([text])
        prediction = model.predict(text_vector)[0]
        probability = model.predict_proba(text_vector)[0]
        
        label = "🤖 Computer Generated (Fake)" if prediction == 1 else "✅ Original (Real)"
        confidence = max(probability) * 100
        
        return f"{label} (Confidence: {confidence:.1f}%)"
        
    except Exception as e:
        return f"❌ Error with TF-IDF prediction: {str(e)}"

def predict_with_bert(text):
    """Predict using the BERT transformer model."""
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        
        # Look for BERT model in checkpoints
        model_path = os.path.join(project_root, 'models', 'saved_models', 'bert_checkpoints')
        checkpoint_dirs = []
        
        if os.path.exists(model_path):
            checkpoint_dirs = [d for d in os.listdir(model_path) if d.startswith('checkpoint-')]
            
        if not checkpoint_dirs:
            return "❌ BERT model not found. Please train the BERT model first."
        
        # Use the latest checkpoint
        latest_checkpoint = sorted(checkpoint_dirs, key=lambda x: int(x.split('-')[1]))[-1]
        full_model_path = os.path.join(model_path, latest_checkpoint)
        
        # Load model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(full_model_path)
        model = AutoModelForSequenceClassification.from_pretrained(full_model_path)
        
        # Tokenize and predict
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_class = torch.argmax(predictions, dim=-1).item()
            confidence = predictions[0][predicted_class].item() * 100
        
        label = "🤖 Computer Generated (Fake)" if predicted_class == 1 else "✅ Original (Real)"
        return f"{label} (Confidence: {confidence:.1f}%)"
        
    except Exception as e:
        return f"❌ Error with BERT prediction: {str(e)}"

def main():
    parser = argparse.ArgumentParser(description="Predict if a review is fake or real")
    parser.add_argument("text", help="The review text to analyze")
    parser.add_argument("--model", choices=["tfidf", "bert", "both"], default="both",
                       help="Which model to use for prediction (default: both)")
    
    args = parser.parse_args()
    
    print(f"🔍 Analyzing review: '{args.text}'\n")
    
    if args.model in ["tfidf", "both"]:
        print("📊 TF-IDF + Logistic Regression:")
        result = predict_with_tfidf(args.text)
        print(f"   {result}\n")
    
    if args.model in ["bert", "both"]:
        print("🧠 BERT Transformer:")
        result = predict_with_bert(args.text)
        print(f"   {result}\n")

if __name__ == "__main__":
    main()
