# 🚀 L1 Super Makkelijk Dataset Systeem

## 🎯 Overzicht

Met dit systeem kun je **super makkelijk** nieuwe datasets toevoegen aan je L1 LLM! Alles wordt automatisch gedetecteerd en verwerkt.

## 📊 Beschikbare Datasets

### Quick Start
```bash
# Bekijk alle beschikbare datasets
python dataset_manager.py --list

# Gebruik een preset (aanbevolen voor beginners)
python dataset_manager.py --preset beginner

# Setup specifieke dataset
python dataset_manager.py --setup wikipedia_simple
```

### 🏆 Aanbevolen Datasets

| Dataset | Gebruik | Kwaliteit | Samples |
|---------|---------|-----------|---------|
| `wikipedia_simple` | Algemene kennis | ⭐⭐⭐⭐⭐ | 100k |
| `news_all` | Actuele events | ⭐⭐⭐⭐ | 50k |
| `books_gutenberg` | Literatuur | ⭐⭐⭐⭐⭐ | 80k |
| `code_github` | Programming | ⭐⭐⭐ | 150k |

## 🆕 Nieuwe Dataset Toevoegen

### Methode 1: Super Makkelijk (Interactief)
```bash
python add_dataset.py --interactive
```
Dit vraagt je stap voor stap alles wat nodig is!

### Methode 2: Via Command Line
```bash
python add_dataset.py \
  --dataset-id my_cool_dataset \
  --name "Mijn Coole Dataset" \
  --description "Beschrijving van mijn dataset" \
  --method kagglehub \
  --path "username/dataset-name" \
  --samples 50000
```

### Methode 3: Direct in YAML File
Voeg gewoon toe aan `datasets.yaml`:
```yaml
datasets:
  my_dataset:
    name: "Mijn Dataset"
    description: "Beschrijving"
    download_method: "kagglehub"
    kagglehub_path: "user/dataset-name"
    auto_detect_format: true
    recommended_samples: 100000
    recommended_vocab: 20000
    quality: "high"
    topics: ["custom", "cool"]
```

## 🎯 Presets Gebruiken

Kies een preset gebaseerd op je doel:

```bash
# Voor beginners
python dataset_manager.py --preset beginner

# Voor gevorderde gebruikers  
python dataset_manager.py --preset advanced

# Voor conversational AI
python dataset_manager.py --preset conversational

# Voor technische/code AI
python dataset_manager.py --preset technical
```

## 🔍 Datasets Zoeken en Filteren

```bash
# Filter op onderwerp
python dataset_manager.py --list --filter-topic programming

# Filter op kwaliteit
python dataset_manager.py --list --filter-quality very_high

# Bekijk alle presets
python dataset_manager.py --list-presets
```

## 🛠️ Download Methodes

### 1. KaggleHub (Modern, Aanbevolen)
```yaml
download_method: "kagglehub"
kagglehub_path: "username/dataset-name"
```
- ✅ Sneller dan traditionele Kaggle API
- ✅ Automatische caching
- ✅ Betere error handling

### 2. Kaggle API (Traditioneel)
```yaml
download_method: "kaggle_api" 
kaggle_path: "username/dataset-name"
```
- ✅ Werkt met alle Kaggle datasets
- ⚠️ Vereist Kaggle API setup

### 3. Custom (Voor eigen datasets)
```yaml
download_method: "custom"
custom_path: "/path/to/dataset"
```
- ✅ Voor lokale bestanden
- ✅ Voor eigen download scripts

## 📁 File Formaten

Het systeem detecteert automatisch:
- ✅ CSV bestanden (`.csv`)
- ✅ JSON bestanden (`.json`)
- ✅ JSONL bestanden (`.jsonl`)  
- ✅ Text bestanden (`.txt`)

### Kolom Namen
Het systeem probeert automatisch deze kolom namen:
- `text`, `content`, `body`, `article`, `description`

## 🎨 Complete Workflow

### 1. Dataset Toevoegen
```bash
# Voeg je dataset toe
python add_dataset.py --interactive
```

### 2. Dataset Setup
```bash
# Download en verwerk automatisch
python dataset_manager.py --setup your_dataset_name
```

### 3. Training Starten
```bash
# Start GPU training
python train_gpu.py

# Of CPU training
python train_minimal.py
```

## 📊 Voorbeeld: Nieuwe Dataset Toevoegen

Stel je hebt een nieuwe Kaggle dataset gevonden:

```bash
# Stap 1: Voeg toe via interactief script
python add_dataset.py -i

# Het script vraagt:
# Dataset ID: my_news_dataset
# Naam: Latest News Articles
# Beschrijving: Recent news from multiple sources
# Download methode: 1 (kagglehub)
# Kagglehub pad: newscorp/latest-articles
# Auto-detect: y
# Samples: 75000
# Vocab: 18000
# Kwaliteit: 3 (high)
# Topics: news,current

# Stap 2: Test de dataset
python dataset_manager.py --setup my_news_dataset

# Stap 3: Start training
python train_gpu.py
```

## 🔧 Troubleshooting

### Dataset Download Faalt
```bash
# Check je Kaggle credentials
kaggle datasets list

# Probeer handmatige download
python dataset_manager.py --setup dataset_name --samples 10000
```

### File Format Issues
```bash
# Bekijk welke bestanden gevonden zijn
ls -la datasets/your_dataset/

# Probeer handmatige processing
python prepare_large_dataset.py "path/to/file.csv" --text-column "content"
```

### Geheugen Problemen
```bash
# Gebruik minder samples
python dataset_manager.py --setup dataset_name --samples 50000 --vocab-size 10000
```

## 🎉 Voordelen van Dit Systeem

- ✅ **Een command**: Download + verwerk + klaar voor training
- ✅ **Auto-detectie**: Formaten en kolommen automatisch gevonden
- ✅ **Flexibel**: Makkelijk nieuwe datasets toevoegen
- ✅ **Presets**: Voorgedefinieerde configuraties
- ✅ **Error handling**: Probeert verschillende opties als iets faalt
- ✅ **Overzichtelijk**: Alles in één YAML bestand

Nu kun je **elke dataset** die je vindt binnen 2 minuten gebruiken voor je L1 training! 🚀
