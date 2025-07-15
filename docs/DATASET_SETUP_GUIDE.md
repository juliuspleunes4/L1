# 📚 Dataset Setup Guide for L1

## 1. Install Kaggle API
```bash
pip install kaggle
```

## 2. Setup Kaggle Credentials
1. Go to kaggle.com → Account → Create New API Token
2. Download `kaggle.json` 
3. Place it in: `C:\Users\{username}\.kaggle\kaggle.json` (Windows)
4. Or set environment variables:
   ```bash
   set KAGGLE_USERNAME=your_username
   set KAGGLE_KEY=your_api_key
   ```

## 3. Popular Text Datasets for LLM Training

### News Articles (Great for general knowledge)
```bash
# All The News (143k articles)
kaggle datasets download -d snapcrack/all-the-news -p ./datasets/
unzip ./datasets/all-the-news.zip -d ./datasets/news/

# Process the dataset
python prepare_large_dataset.py "datasets/news/articles1.csv" --text-column "content" --max-samples 100000 --vocab-size 20000
```

### Wikipedia Text (High quality, diverse topics)
```bash
# Wikipedia Text (6M articles)
kaggle datasets download -d jkkphys/english-wikipedia-articles-20170820-sqlite -p ./datasets/
# This is SQLite format - you'll need to extract text

# Alternative: Simple Wikipedia
kaggle datasets download -d mikeortman/wikipedia-sentences -p ./datasets/
python prepare_large_dataset.py "datasets/wikipedia-sentences.csv" --text-column "sentence"
```

### Books/Literature (Rich language patterns)
```bash
# Project Gutenberg Books
kaggle datasets download -d alexandreparent/gutenberg-database -p ./datasets/

# BookCorpus (larger dataset)
kaggle datasets download -d vishnurapps/book-corpus -p ./datasets/
```

### Code/Programming (If you want code-aware model)
```bash
# GitHub repositories
kaggle datasets download -d github/github-repos -p ./datasets/

# Stack Overflow
kaggle datasets download -d stackoverflow/stackoverflow -p ./datasets/
```

### Reddit/Social Media (Conversational patterns)
```bash
# Reddit comments
kaggle datasets download -d reddit/reddit-comments-2017 -p ./datasets/

# Twitter sentiment
kaggle datasets download -d kazanova/sentiment140 -p ./datasets/
```

## 4. Example: Quick Start with News Dataset

### Download and Process
```bash
# 1. Download news dataset (smaller, good for testing)
kaggle datasets download -d snapcrack/all-the-news -p ./datasets/

# 2. Extract
cd datasets
unzip all-the-news.zip

# 3. Go back to L1 directory
cd ..

# 4. Process the dataset
python prepare_large_dataset.py "datasets/articles1.csv" --text-column "content" --max-samples 50000 --vocab-size 15000
```

### Expected Output
```
📊 Dataset Statistics:
   ├── Total samples: 50,000
   ├── Average length: 2,847.3 characters
   ├── Min length: 287 characters
   └── Max length: 15,423 characters

✅ Dataset saved:
   ├── Training: 45,000 samples → data/processed/train.txt
   ├── Validation: 5,000 samples → data/processed/val.txt
   └── Vocabulary: 15,000 tokens → data/processed/tokenizer.json
```

## 5. Advanced: Combine Multiple Datasets

### Script to merge datasets
```python
# combine_datasets.py
import pandas as pd
import os

datasets = [
    ("datasets/news/articles1.csv", "content"),
    ("datasets/books/books.csv", "text"), 
    ("datasets/wiki/wikipedia.csv", "text")
]

all_texts = []
for file_path, column in datasets:
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        texts = df[column].dropna().tolist()
        all_texts.extend(texts)
        print(f"Added {len(texts):,} texts from {file_path}")

# Save combined dataset
with open("datasets/combined_dataset.txt", "w", encoding="utf-8") as f:
    for text in all_texts:
        if isinstance(text, str) and len(text) > 50:
            f.write(text.strip() + "\n")

print(f"Combined dataset: {len(all_texts):,} total texts")
```

## 6. Dataset Size Recommendations

### For Testing (Your laptop)
- **10K-100K samples**: Quick iteration and testing
- **Vocab size**: 5K-10K tokens
- **Model**: Keep current size (19M params)

### For GPU Training (Your new PC)
- **100K-1M samples**: Good performance gains
- **Vocab size**: 15K-50K tokens  
- **Model**: Use GPU config (80M+ params)

### For Production Training
- **1M+ samples**: State-of-the-art results
- **Vocab size**: 50K+ tokens
- **Model**: Scale up further

## 7. Storage Requirements

| Dataset Size | Disk Space | Processing Time | RAM Needed |
|--------------|------------|-----------------|------------|
| 50K samples  | ~500MB     | 5-10 minutes    | 4GB        |
| 500K samples | ~5GB       | 30-60 minutes   | 8GB        |
| 5M samples   | ~50GB      | 3-6 hours       | 16GB+      |

## 8. Troubleshooting

### Common Issues:
```bash
# If Kaggle download fails
kaggle datasets download -d snapcrack/all-the-news --force

# If processing runs out of memory
python prepare_large_dataset.py dataset.csv --max-samples 100000  # Limit samples

# If vocabulary is too large
python prepare_large_dataset.py dataset.csv --vocab-size 10000  # Smaller vocab
```
