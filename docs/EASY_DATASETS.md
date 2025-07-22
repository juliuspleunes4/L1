# 🚀 L1 Easy Dataset System

## 🎯 Overview

L1 includes a powerful dataset management system that makes adding new datasets incredibly easy. Everything is automatically detected and processed.

## 📊 Current Status

✅ **Your L1 is ready with Wikipedia Simple dataset:**
- **90,000+ high-quality samples** already processed
- **BPE tokenizer** with 1,783 vocabulary pre-trained  
- **Training data** ready in `data/processed/`

## 🆕 Quick Dataset Options

### Option 1: Use Pre-configured Presets
```bash
# Beginner: 50k samples, fast training
python add_dataset.py --preset beginner

# Intermediate: 150k samples, balanced  
python add_dataset.py --preset intermediate

# Advanced: 500k samples, comprehensive
python add_dataset.py --preset advanced
```

### Option 2: Choose Specific Datasets
```bash
# High-quality datasets ready to use
python add_dataset.py wikipedia_full        # 500k Wikipedia articles
python add_dataset.py papers_arxiv          # Scientific papers
python add_dataset.py books_gutenberg       # Classic literature
python add_dataset.py openwebtext           # GPT-style web text

# Specialized datasets
python add_dataset.py code_stackoverflow    # Programming Q&A
python add_dataset.py reddit_comments       # Conversational text  
python add_dataset.py news_all              # Current events
```

### Option 3: Continue with Current Dataset
```bash
# Your current dataset is excellent - continue training!
python train_gpu_compatible.py
```

## 🏆 Available Dataset Categories

| Category | Best Datasets | Samples | Quality | Use Case |
|----------|---------------|---------|---------|----------|
| **Education** | wikipedia_simple ✅ | 90k | ⭐⭐⭐⭐⭐ | **Current** |
| **General** | wikipedia_full, openwebtext | 500k | ⭐⭐⭐⭐⭐ | Production |
| **Technical** | papers_arxiv, code_stackoverflow | 250k | ⭐⭐⭐⭐⭐ | Code/Science |
| **Creative** | books_gutenberg, stories | 80k | ⭐⭐⭐⭐⭐ | Writing |
| **Conversational** | reddit_comments, chat | 200k | ⭐⭐⭐⭐ | Dialogue |

## 🆕 Adding Your Own Kaggle Dataset

### Method 1: KaggleHub (Easiest)
```python
import kagglehub
# Download any Kaggle dataset directly
dataset_path = kagglehub.dataset_download("username/dataset-name")
python prepare_large_dataset.py --custom_path dataset_path
```

### Method 2: Add to datasets.yaml (Permanent)
Edit `datasets.yaml` file:
```yaml
your_awesome_dataset:
  name: "Your Awesome Dataset"
  description: "Description of your dataset"
  download_method: "kagglehub"
  kagglehub_path: "username/dataset-name"  # From Kaggle URL
  auto_detect_format: true
  recommended_samples: 100000
  recommended_vocab: 20000
  quality: "high"
  topics: ["custom", "awesome"]
```

Then use it:
```bash
python add_dataset.py your_awesome_dataset
```

### Method 3: Command Line
```bash
# For quick testing without editing files
python add_dataset.py --custom /path/to/dataset
```

## 🎯 Dataset Presets Explained

### Beginner Preset
- **Datasets**: Wikipedia Simple + News
- **Samples**: 50,000 (fast training)
- **Vocab Size**: 15,000 tokens
- **Training Time**: 2-4 hours on RTX 5060 Ti
- **Use Case**: Learning, quick experiments

### Intermediate Preset  
- **Datasets**: Wikipedia + Books + News
- **Samples**: 150,000 (balanced performance)
- **Vocab Size**: 25,000 tokens
- **Training Time**: 6-12 hours on RTX 5060 Ti
- **Use Case**: Good model performance

### Advanced Preset
- **Datasets**: Wikipedia + Papers + Books
- **Samples**: 500,000 (high quality)
- **Vocab Size**: 50,000 tokens  
- **Training Time**: 24-48 hours on RTX 5060 Ti
- **Use Case**: Production-ready models

### Specialized Presets
```bash
python add_dataset.py --preset conversational  # Reddit + Twitter + Chat
python add_dataset.py --preset technical       # GitHub + Stack Overflow
python add_dataset.py --preset knowledge       # Full Wikipedia + Papers  
python add_dataset.py --preset creative        # Books + Literature + Stories
```

## 🛠️ Dataset Management Commands

### View Available Options
```bash
# List all pre-configured datasets
python dataset_manager.py --list

# Get detailed info about a specific dataset  
python dataset_manager.py --info wikipedia_simple

# Preview dataset samples
python dataset_manager.py --preview wikipedia_simple --samples 5

# Check current dataset status
python dataset_manager.py --stats
```

### Verify Current Setup
```bash
# Check if your current data is ready
ls data/processed/
# Should show: train.txt, val.txt, tokenizer.json

# View current dataset statistics
python -c "
with open('data/processed/train.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()
    print(f'Training samples: {len(lines):,}')
    print('✅ Ready for training!')
"
```

## 📁 Supported File Formats

L1 automatically detects and processes:
- ✅ **CSV files** (`.csv`) - Most common format
- ✅ **JSON files** (`.json`) - Structured data
- ✅ **JSONL files** (`.jsonl`) - Line-separated JSON
- ✅ **Text files** (`.txt`) - Plain text datasets

### Automatic Column Detection
The system automatically finds text in these column names:
- `text`, `content`, `body`, `article`, `description`, `message`, `comment`

## 🎨 Complete Workflow Example

### Scenario: Adding a Custom News Dataset
```bash
# Step 1: Find dataset on Kaggle
# Go to kaggle.com, find "awesome-news-dataset" by "newscompany"

# Step 2: Add to L1 (easiest method)
python -c "
import kagglehub
dataset_path = kagglehub.dataset_download('newscompany/awesome-news-dataset')
print(f'Downloaded to: {dataset_path}')
"

# Step 3: Process with L1
python prepare_large_dataset.py --custom_path /path/to/dataset

# Step 4: Start training
python train_gpu_compatible.py
```

### Alternative: Add Permanently to datasets.yaml
```bash
# Edit datasets.yaml to add:
awesome_news:
  name: "Awesome News Dataset" 
  description: "High-quality news articles from multiple sources"
  download_method: "kagglehub"
  kagglehub_path: "newscompany/awesome-news-dataset"
  auto_detect_format: true
  recommended_samples: 75000
  recommended_vocab: 20000
  quality: "high"
  topics: ["news", "current_events"]

# Then use it:
python add_dataset.py awesome_news
```

## 🔧 Troubleshooting

### Dataset Download Issues
```bash
# Check your internet connection and try again
python add_dataset.py your_dataset --retry

# For Kaggle datasets, verify the path is correct
# URL: https://www.kaggle.com/datasets/username/dataset-name
# Use: username/dataset-name
```

### File Format Issues
```bash
# Check what files were downloaded
ls -la data/raw/your_dataset/

# Try manual processing with specific column
python prepare_large_dataset.py "data/raw/dataset.csv" --text-column "your_column_name"
```

### Memory Issues
```bash
# Use fewer samples for testing
python add_dataset.py your_dataset --max-samples 10000

# Use smaller vocabulary
python prepare_large_dataset.py dataset.csv --vocab-size 5000
```

### Training Issues After Dataset Change
```bash
# Clear old model data when switching datasets
rm -rf models/l1-gpu-compatible/checkpoint_*.pt

# Start fresh training
python train_gpu_compatible.py
```

## ✅ Current Recommendation

**For immediate use**: Keep your current Wikipedia Simple dataset
- ✅ **High-quality** educational content (90k+ samples)
- ✅ **Already processed** and ready for training
- ✅ **Perfect for development** and learning L1
- ✅ **Good vocabulary coverage** for general language

**For future experiments**: Try these after completing current training:
1. `python add_dataset.py --preset intermediate` (more diverse content)
2. Increase vocabulary size (1783 → 8000+) for better text quality
3. Specialized datasets based on your specific use case

## 🎉 System Benefits

- ✅ **One Command**: Download + process + ready for training
- ✅ **Auto-Detection**: File formats and text columns automatically found
- ✅ **Flexible**: Easy to add any new dataset
- ✅ **Presets**: Pre-configured combinations for common use cases
- ✅ **Error Handling**: Tries multiple approaches if something fails
- ✅ **Organized**: Everything managed in one `datasets.yaml` file
- ✅ **Production Ready**: Handles large datasets efficiently

With L1's dataset system, you can use **any Kaggle dataset** within 2 minutes for your LLM training! 🚀
