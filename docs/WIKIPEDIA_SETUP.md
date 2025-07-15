# 🌍 Wikipedia Dataset Setup voor L1

## Nieuwe Moderne Methode (Aanbevolen)

Je hebt een uitstekende dataset gevonden! Wikipedia Simple English is perfect voor LLM training.

### 🚀 Snelle Setup

```bash
# Optie 1: Gebruik de nieuwe Wikipedia download script
python download_wikipedia.py --max-samples 100000 --vocab-size 20000

# Optie 2: Gebruik de geïntegreerde setup 
python setup_gpu_training.py --dataset wikipedia --samples 100000
```

### 📊 Dataset Details

**Wikipedia Simple English Dataset:**
- **Bron**: `ffatty/plain-text-wikipedia-simpleenglish`
- **Grootte**: Miljoenen Wikipedia artikelen in eenvoudig Engels
- **Kwaliteit**: Zeer hoog - gestructureerde, feitelijke content
- **Taal**: Eenvoudig Engels (perfect voor training)
- **Download methode**: Moderne `kagglehub` (sneller en betrouwbaarder)

### 🔧 Wat Er Gebeurt

1. **Automatische installatie**: `kagglehub` wordt geïnstalleerd als het ontbreekt
2. **Download**: Dataset wordt gedownload naar lokale cache
3. **Detectie**: Script vindt automatisch de beste tekstbestanden
4. **Verwerking**: Converteert naar L1 formaat (train.txt, val.txt, tokenizer.json)
5. **Klaar voor training**: Direct bruikbaar met je GPU training script

### 💡 Voordelen van Deze Dataset

- **Grote vocabulaire**: Veel verschillende woorden en concepten
- **Goede grammatica**: Wikipedia heeft hoge kwaliteit tekst
- **Diverse onderwerpen**: Van wetenschap tot geschiedenis tot cultuur
- **Consistente stijl**: Encyclopedische schrijfstijl
- **Eenvoudig Engels**: Makkelijker te leren voor je model

### 🎯 Aanbevolen Settings

Voor je nieuwe GPU PC:
```bash
# Grote dataset voor krachtige training
python download_wikipedia.py --max-samples 500000 --vocab-size 30000

# Of voor snelle test eerst
python download_wikipedia.py --max-samples 50000 --vocab-size 15000
```

### 📈 Verwachte Resultaten

Met Wikipedia data krijg je:
- **Betere taalvaardigheid**: Model leert diverse onderwerpen
- **Feitelijke kennis**: Echte informatie over de wereld  
- **Goede grammatica**: Correct taalgebruik
- **Breed vocabulaire**: Veel verschillende woorden

### 🔄 Na Download

```bash
# Check of alles goed is gegaan
ls -la data/processed/
# Je zou moeten zien:
# train.txt (training data)
# val.txt (validation data) 
# tokenizer.json (vocabulary)

# Start training op GPU
python train_gpu.py

# Of op CPU voor test
python train_minimal.py
```

### 🚨 Troubleshooting

Als download faalt:
```bash
# Installeer kagglehub handmatig
pip install kagglehub

# Of gebruik oudere methode
python setup_gpu_training.py --dataset wikipedia-old
```

### 🎉 Perfect Voor Je Use Case!

Deze Wikipedia dataset is ideaal omdat:
- ✅ Groot genoeg voor je GPU training
- ✅ Hoge kwaliteit content
- ✅ Moderne download methode
- ✅ Automatische verwerking
- ✅ Direct compatible met je L1 model

Je kunt nu een echt capabele LLM trainen met echte kennis! 🧠🚀
