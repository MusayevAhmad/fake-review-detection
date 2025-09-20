#!/usr/bin/env python3
"""
Setup script for Fake Review Detection project.
This script helps users get started by downloading NLTK data and checking dependencies.
"""

import sys
import subprocess
import os

def install_requirements():
    """Install required packages from requirements.txt"""
    print("📦 Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Packages installed successfully!")
    except subprocess.CalledProcessError:
        print("❌ Failed to install packages. Please install manually using:")
        print("   pip install -r requirements.txt")
        return False
    return True

def download_nltk_data():
    """Download required NLTK data"""
    print("\n📚 Downloading NLTK data...")
    import nltk
    
    required_data = [
        'punkt',
        'stopwords', 
        'wordnet',
        'averaged_perceptron_tagger',
        'omw-1.4'
    ]
    
    for data in required_data:
        try:
            print(f"   Downloading {data}...")
            nltk.download(data, quiet=True)
        except Exception as e:
            print(f"   ⚠️  Could not download {data}: {e}")
    
    print("✅ NLTK data download complete!")

def check_data_files():
    """Check if data files exist"""
    print("\n📂 Checking data files...")
    data_dir = "data"
    
    if not os.path.exists(data_dir):
        print(f"❌ Data directory '{data_dir}' not found!")
        return False
    
    required_files = [
        "fake_reviews_dataset.csv",
        "processed_reviews.json",
        "test_nltk.csv"
    ]
    
    missing_files = []
    for file in required_files:
        file_path = os.path.join(data_dir, file)
        if os.path.exists(file_path):
            print(f"   ✅ {file}")
        else:
            print(f"   ❌ {file} (missing)")
            missing_files.append(file)
    
    if missing_files:
        print(f"\n⚠️  Missing {len(missing_files)} data file(s). Please ensure all data files are in the 'data/' directory.")
        return False
    
    print("✅ All data files found!")
    return True

def create_directories():
    """Create necessary directories if they don't exist"""
    print("\n📁 Creating directories...")
    
    directories = [
        "models/saved_models",
        "results",
        "logs"
    ]
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"   Created: {directory}")
        else:
            print(f"   Exists: {directory}")
    
    print("✅ Directory structure ready!")

def main():
    """Main setup function"""
    print("🚀 Fake Review Detection - Setup Script")
    print("=" * 50)
    
    # Install requirements
    if not install_requirements():
        sys.exit(1)
    
    # Download NLTK data
    try:
        download_nltk_data()
    except ImportError:
        print("❌ NLTK not installed. Please run: pip install -r requirements.txt")
        sys.exit(1)
    
    # Check data files
    check_data_files()
    
    # Create directories
    create_directories()
    
    print("\n🎉 Setup complete!")
    print("\nNext steps:")
    print("1. Train models: python models/traditional_ml/tfidf_ngram_train.py")
    print("2. Run demo: python demo/predict_review.py 'Your review text here'")
    print("3. GUI demo: python demo/gui_demo.py")
    print("\nFor more information, see README.md")

if __name__ == "__main__":
    main()
