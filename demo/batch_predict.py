#!/usr/bin/env python3
"""
Batch prediction script for processing multiple reviews.

Usage:
    python demo/batch_predict.py input_file.txt output_file.csv
    python demo/batch_predict.py --model bert reviews.txt results.csv
"""

import argparse
import csv
import sys
import os

# Add the project root to the path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

def load_tfidf_model():
    """Load TF-IDF model and vectorizer"""
    try:
        import joblib
        model_path = os.path.join(project_root, 'models', 'saved_models', 'best_logistic_regression_ngram.joblib')
        vectorizer_path = os.path.join(project_root, 'models', 'saved_models', 'tfidf_ngram_vectorizer.joblib')
        
        if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
            return None, None
        
        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
        return model, vectorizer
    except Exception as e:
        print(f"Error loading TF-IDF model: {e}")
        return None, None

def load_bert_model():
    """Load BERT model and tokenizer"""
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        
        model_path = os.path.join(project_root, 'models', 'saved_models', 'bert_checkpoints')
        checkpoint_dirs = []
        
        if os.path.exists(model_path):
            checkpoint_dirs = [d for d in os.listdir(model_path) if d.startswith('checkpoint-')]
            
        if not checkpoint_dirs:
            return None, None
        
        # Use the latest checkpoint
        latest_checkpoint = sorted(checkpoint_dirs, key=lambda x: int(x.split('-')[1]))[-1]
        full_model_path = os.path.join(model_path, latest_checkpoint)
        
        tokenizer = AutoTokenizer.from_pretrained(full_model_path)
        model = AutoModelForSequenceClassification.from_pretrained(full_model_path)
        return model, tokenizer
    except Exception as e:
        print(f"Error loading BERT model: {e}")
        return None, None

def predict_batch_tfidf(texts, model, vectorizer):
    """Predict using TF-IDF model"""
    text_vectors = vectorizer.transform(texts)
    predictions = model.predict(text_vectors)
    probabilities = model.predict_proba(text_vectors)
    
    results = []
    for i, pred in enumerate(predictions):
        label = "Computer Generated" if pred == 1 else "Original"
        confidence = max(probabilities[i]) * 100
        results.append((label, confidence))
    
    return results

def predict_batch_bert(texts, model, tokenizer):
    """Predict using BERT model"""
    import torch
    
    results = []
    
    # Process in batches to avoid memory issues
    batch_size = 16
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        
        inputs = tokenizer(batch_texts, return_tensors="pt", truncation=True, 
                          padding=True, max_length=512)
        
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_classes = torch.argmax(predictions, dim=-1)
            
            for j, pred_class in enumerate(predicted_classes):
                label = "Computer Generated" if pred_class.item() == 1 else "Original"
                confidence = predictions[j][pred_class].item() * 100
                results.append((label, confidence))
    
    return results

def read_input_file(file_path):
    """Read reviews from input file (one per line)"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]
        return texts
    except Exception as e:
        print(f"Error reading input file: {e}")
        return None

def write_results(output_path, texts, results, model_name):
    """Write results to CSV file"""
    try:
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Review', 'Prediction', 'Confidence', 'Model'])
            
            for text, (label, confidence) in zip(texts, results):
                writer.writerow([text, label, f"{confidence:.2f}%", model_name])
        
        print(f"✅ Results saved to {output_path}")
    except Exception as e:
        print(f"Error writing results: {e}")

def main():
    parser = argparse.ArgumentParser(description="Batch prediction for fake review detection")
    parser.add_argument("input_file", help="Input file with reviews (one per line)")
    parser.add_argument("output_file", help="Output CSV file for results")
    parser.add_argument("--model", choices=["tfidf", "bert"], default="tfidf",
                       help="Which model to use (default: tfidf)")
    
    args = parser.parse_args()
    
    # Read input texts
    texts = read_input_file(args.input_file)
    if texts is None:
        sys.exit(1)
    
    print(f"📄 Processing {len(texts)} reviews using {args.model.upper()} model...")
    
    # Load model and predict
    if args.model == "tfidf":
        model, vectorizer = load_tfidf_model()
        if model is None:
            print("❌ Could not load TF-IDF model. Please train it first.")
            sys.exit(1)
        results = predict_batch_tfidf(texts, model, vectorizer)
        model_name = "TF-IDF + Logistic Regression"
    
    else:  # BERT
        model, tokenizer = load_bert_model()
        if model is None:
            print("❌ Could not load BERT model. Please train it first.")
            sys.exit(1)
        results = predict_batch_bert(texts, model, tokenizer)
        model_name = "BERT Transformer"
    
    # Write results
    write_results(args.output_file, texts, results, model_name)
    
    # Summary
    fake_count = sum(1 for label, _ in results if label == "Computer Generated")
    real_count = len(results) - fake_count
    
    print(f"\n📊 Summary:")
    print(f"   Real reviews: {real_count} ({real_count/len(results)*100:.1f}%)")
    print(f"   Fake reviews: {fake_count} ({fake_count/len(results)*100:.1f}%)")

if __name__ == "__main__":
    main()
