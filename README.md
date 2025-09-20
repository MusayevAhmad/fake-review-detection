# 🕵️ Fake Review Detection System

[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12%2B-orange)](https://pytorch.org/)
[![Scikit-learn](https://img.shields.io/badge/scikit--learn-1.2%2B-red)](https://scikit-learn.org/)
[![Transformers](https://img.shields.io/badge/🤗%20Transformers-4.21%2B-yellow)](https://huggingface.co/transformers/)

A comprehensive machine learning system for detecting fake reviews using both traditional ML approaches and state-of-the-art deep learning models. This project implements multiple methodologies to identify computer-generated (CG) vs. original (OR) reviews with high accuracy.

## 📋 Table of Contents

- [🎯 Project Overview](#-project-overview)
- [🚀 Features](#-features)
- [📊 Models Implemented](#-models-implemented)
- [📁 Repository Structure](#-repository-structure)
- [⚙️ Installation](#️-installation)
- [🔧 Usage](#-usage)
- [📈 Results](#-results)
- [🎮 Demo Applications](#-demo-applications)
- [📝 Documentation](#-documentation)
- [🤝 Contributing](#-contributing)

## 🎯 Project Overview

This project addresses the growing problem of fake reviews in e-commerce platforms by implementing and comparing multiple machine learning approaches:

- **Traditional ML**: TF-IDF vectorization with Logistic Regression and Support Vector Machines
- **Advanced Features**: N-gram analysis (unigrams, bigrams, trigrams) and Part-of-Speech (POS) tagging
- **Deep Learning**: Fine-tuned BERT transformer model for sequence classification
- **Comprehensive Evaluation**: Cross-validation, grid search, and detailed performance metrics

## 🚀 Features

- ✅ **Multiple Model Architectures**: Traditional ML and Transformer-based approaches
- ✅ **Advanced Text Processing**: N-grams, POS tagging, and feature engineering
- ✅ **Hyperparameter Optimization**: Grid search with cross-validation
- ✅ **Model Persistence**: Save and load trained models
- ✅ **Visualization**: Confusion matrices, training metrics, and performance plots
- ✅ **Interactive Demo**: GUI application for real-time predictions
- ✅ **Comprehensive Evaluation**: Train/validation/test splits with detailed metrics

## 📊 Models Implemented

### 1. TF-IDF + Logistic Regression
- **Features**: Unigrams, bigrams, trigrams
- **Enhancement**: POS tagging integration
- **Optimization**: Grid search for hyperparameters
- **Performance**: ~90%+ accuracy on validation set

### 2. Support Vector Machine (SVM)
- **Kernels**: Linear and RBF kernels
- **Features**: N-gram TF-IDF with POS tags
- **Optimization**: Comprehensive grid search
- **Performance**: Competitive with logistic regression

### 3. BERT Transformer
- **Model**: Pre-trained BERT-base-uncased
- **Fine-tuning**: Task-specific classification head
- **Training**: Custom training loop with validation
- **Performance**: State-of-the-art results

## 📁 Repository Structure

```
📦 fake-review-detection/
├── 📂 data/                          # Dataset files
│   ├── fake_reviews_dataset.csv      # Main dataset
│   ├── processed_reviews.json        # Processed data
│   └── test_nltk.csv                 # Test data
├── 📂 models/                        # Model implementations
│   ├── 📂 traditional_ml/            # TF-IDF & SVM models
│   │   ├── tfidf_ngram_train.py      # TF-IDF training
│   │   ├── tfidf_ngram_POS_train.py  # TF-IDF with POS
│   │   ├── svm_ngram_pos_train.py    # SVM training
│   │   └── svm_ngram_pos_gridsearch.py # Hyperparameter tuning
│   ├── 📂 deep_learning/             # BERT implementations
│   │   ├── bert_semih.py             # BERT training script
│   │   └── bert_test.py              # BERT evaluation
│   └── 📂 saved_models/              # Trained models
├── 📂 scripts/                       # Utility scripts
│   ├── word_counter.py               # Text analysis utilities
│   ├── tokenization.py              # Text preprocessing
│   └── tfidf_analysis.py            # Feature analysis
├── 📂 notebooks/                     # Jupyter notebooks
│   └── data_exploration.ipynb        # Data analysis & visualization
├── 📂 demo/                          # Demo applications
│   ├── gui_demo.py                   # Tkinter GUI application
│   └── single_review_inference_demo.py # Command-line demo
├── 📂 results/                       # Training results & plots
│   ├── confusion_matrix_*.png        # Model performance visualizations
│   └── *_results.txt                # Detailed metrics
├── 📂 docs/                          # Documentation
│   └── project_report.pdf            # Comprehensive project report
├── requirements.txt                   # Python dependencies
├── README.md                         # This file
└── .gitignore                        # Git ignore rules
```

## ⚙️ Installation

### 1. Clone the Repository
```bash
git clone https://github.com/MusayevAhmad/MachineLearningProject.git
cd MachineLearningProject
```

### 2. Create Virtual Environment
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Download NLTK Data
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
```

## 🔧 Usage

### Training Models

#### 1. Traditional ML Models
```bash
# Train TF-IDF + Logistic Regression
python models/traditional_ml/tfidf_ngram_train.py

# Train TF-IDF with POS tagging
python models/traditional_ml/tfidf_ngram_POS_train.py

# Train SVM with hyperparameter tuning
python models/traditional_ml/svm_ngram_pos_gridsearch.py
```

#### 2. BERT Model
```bash
# Train BERT model
python models/deep_learning/bert_semih.py
```

### Model Evaluation
```bash
# Evaluate TF-IDF model
python models/traditional_ml/tfidf_ngram_test.py

# Evaluate SVM model
python models/traditional_ml/svm_ngram_pos_test.py

# Evaluate BERT model
python models/deep_learning/bert_test.py
```

### Running Demos
```bash
# GUI Demo Application
python demo/gui_demo.py

# Command-line inference
python demo/single_review_inference_demo.py
```

## 📈 Results

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|---------|----------|
| TF-IDF + LogReg | 90.2% | 89.8% | 90.5% | 90.1% |
| TF-IDF + POS + LogReg | 91.5% | 91.2% | 91.8% | 91.5% |
| SVM (RBF) | 89.8% | 89.3% | 90.2% | 89.7% |
| SVM + POS | 92.1% | 91.9% | 92.3% | 92.1% |
| BERT | **93.7%** | **93.4%** | **94.0%** | **93.7%** |

*Results on test set. BERT achieves state-of-the-art performance.*

## 🎮 Demo Applications

### GUI Application
- **File**: `demo/gui_demo.py`
- **Features**: User-friendly interface for single review prediction
- **Usage**: Enter review text and get instant classification results

### Command-line Demo
- **File**: `demo/single_review_inference_demo.py`
- **Features**: Batch processing and programmatic inference
- **Usage**: Process reviews from files or command-line input

## 📝 Documentation

- **Project Report**: Comprehensive analysis in `docs/project_report.pdf`
- **Code Documentation**: Inline comments and docstrings throughout codebase
- **Jupyter Notebooks**: Data exploration and visualization in `notebooks/`

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

- **Group 86** - Machine Learning Project
- **Course**: VU Machine Learning
- **Year**: 2024-2025

## 🙏 Acknowledgments

- VU Faculty for providing the dataset and project guidelines
- Hugging Face for the pre-trained BERT models
- Scikit-learn community for excellent ML tools
- NLTK team for comprehensive NLP utilities

---

⭐ **Star this repository if you found it helpful!** ⭐

For questions or issues, please open an issue on GitHub or contact the authors.
