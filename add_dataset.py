#!/usr/bin/env python3
"""
Quick dataset toevoegen script voor L1
Maakt het super makkelijk om nieuwe datasets toe te voegen!
"""

import yaml
import argparse
from pathlib import Path

def add_dataset_interactive():
    """Interactief dataset toevoegen"""
    print("🆕 Nieuwe Dataset Toevoegen")
    print("=" * 40)
    
    # Basis informatie
    dataset_id = input("📝 Dataset ID (bijv. 'my_dataset'): ").strip()
    if not dataset_id:
        print("❌ Dataset ID is verplicht!")
        return
    
    name = input("📌 Dataset naam: ").strip()
    description = input("📄 Beschrijving: ").strip()
    
    # Download methode
    print("\n🔄 Download methode:")
    print("1. kagglehub (modern, aanbevolen)")
    print("2. kaggle_api (traditioneel)")
    print("3. custom (eigen download)")
    
    method_choice = input("Kies (1/2/3): ").strip()
    method_map = {'1': 'kagglehub', '2': 'kaggle_api', '3': 'custom'}
    download_method = method_map.get(method_choice, 'kagglehub')
    
    # Download pad
    if download_method == 'kagglehub':
        download_path = input("🔗 Kagglehub pad (bijv. 'user/dataset-name'): ").strip()
        path_key = 'kagglehub_path'
    elif download_method == 'kaggle_api':
        download_path = input("🔗 Kaggle API pad (bijv. 'user/dataset-name'): ").strip() 
        path_key = 'kaggle_path'
    else:
        download_path = input("🔗 Custom download URL/pad: ").strip()
        path_key = 'custom_path'
    
    # File configuratie
    print("\n📁 File configuratie:")
    auto_detect = input("🔍 Auto-detect bestand formaat? (y/n) [y]: ").strip().lower()
    auto_detect = auto_detect != 'n'
    
    text_column = None
    file_pattern = None
    
    if not auto_detect:
        file_pattern = input("📋 File pattern (bijv. '*.csv'): ").strip() or None
        text_column = input("📝 Text kolom naam (bijv. 'content'): ").strip() or None
    
    # Aanbevelingen
    print("\n🎯 Aanbevolen waardes:")
    try:
        recommended_samples = int(input("📊 Aanbevolen samples [100000]: ").strip() or "100000")
        recommended_vocab = int(input("📚 Aanbevolen vocab grootte [20000]: ").strip() or "20000")
    except ValueError:
        recommended_samples = 100000
        recommended_vocab = 20000
    
    # Kwaliteit en onderwerpen
    print("\n🏆 Kwaliteit level:")
    print("1. low, 2. medium, 3. high, 4. very_high")
    quality_choice = input("Kies (1/2/3/4) [3]: ").strip() or "3"
    quality_map = {'1': 'low', '2': 'medium', '3': 'high', '4': 'very_high'}
    quality = quality_map.get(quality_choice, 'high')
    
    topics_input = input("🏷️  Topics (comma-separated, bijv. 'news,politics'): ").strip()
    topics = [t.strip() for t in topics_input.split(',') if t.strip()] if topics_input else ["general"]
    
    # Dataset object maken
    dataset_config = {
        'name': name,
        'description': description,
        'download_method': download_method,
        path_key: download_path,
        'auto_detect_format': auto_detect,
        'recommended_samples': recommended_samples,
        'recommended_vocab': recommended_vocab,
        'quality': quality,
        'topics': topics
    }
    
    # Optionele velden
    if file_pattern:
        dataset_config['file_pattern'] = file_pattern
    if text_column:
        dataset_config['text_column'] = text_column
    
    return dataset_id, dataset_config

def load_datasets_config(config_file="datasets.yaml"):
    """Laad bestaande dataset configuratie"""
    if Path(config_file).exists():
        with open(config_file, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    else:
        return {'datasets': {}, 'presets': {}}

def save_datasets_config(config, config_file="datasets.yaml"):
    """Sla dataset configuratie op"""
    with open(config_file, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

def add_dataset_from_args(args):
    """Dataset toevoegen via command line arguments"""
    dataset_config = {
        'name': args.name,
        'description': args.description,
        'download_method': args.method,
        'recommended_samples': args.samples,
        'recommended_vocab': args.vocab_size,
        'quality': args.quality,
        'topics': args.topics.split(',') if args.topics else ['general'],
        'auto_detect_format': not args.no_auto_detect
    }
    
    # Download path
    if args.method == 'kagglehub':
        dataset_config['kagglehub_path'] = args.path
    elif args.method == 'kaggle_api':
        dataset_config['kaggle_path'] = args.path
    else:
        dataset_config['custom_path'] = args.path
    
    # Optionele velden
    if args.file_pattern:
        dataset_config['file_pattern'] = args.file_pattern
    if args.text_column:
        dataset_config['text_column'] = args.text_column
    
    return args.dataset_id, dataset_config

def main():
    parser = argparse.ArgumentParser(description="Voeg nieuwe datasets toe aan L1")
    parser.add_argument("--interactive", "-i", action="store_true", 
                       help="Interactieve modus")
    parser.add_argument("--dataset-id", help="Dataset ID")
    parser.add_argument("--name", help="Dataset naam")
    parser.add_argument("--description", help="Dataset beschrijving")
    parser.add_argument("--method", choices=['kagglehub', 'kaggle_api', 'custom'],
                       default='kagglehub', help="Download methode")
    parser.add_argument("--path", help="Download pad (kagglehub/kaggle path)")
    parser.add_argument("--file-pattern", help="File pattern (bijv. *.csv)")
    parser.add_argument("--text-column", help="Text kolom naam")
    parser.add_argument("--samples", type=int, default=100000,
                       help="Aanbevolen samples")
    parser.add_argument("--vocab-size", type=int, default=20000,
                       help="Aanbevolen vocab grootte")
    parser.add_argument("--quality", choices=['low', 'medium', 'high', 'very_high'],
                       default='high', help="Kwaliteit level")
    parser.add_argument("--topics", help="Topics (comma-separated)")
    parser.add_argument("--no-auto-detect", action="store_true",
                       help="Disable auto format detection")
    parser.add_argument("--config", default="datasets.yaml",
                       help="Config bestand")
    
    args = parser.parse_args()
    
    if args.interactive or not args.dataset_id:
        # Interactieve modus
        dataset_id, dataset_config = add_dataset_interactive()
    else:
        # Command line modus
        if not all([args.dataset_id, args.name, args.description, args.path]):
            print("❌ Verplichte velden: --dataset-id, --name, --description, --path")
            return
        
        dataset_id, dataset_config = add_dataset_from_args(args)
    
    # Laad bestaande configuratie
    config = load_datasets_config(args.config)
    
    # Voeg nieuwe dataset toe
    config['datasets'][dataset_id] = dataset_config
    
    # Sla op
    save_datasets_config(config, args.config)
    
    print(f"\n✅ Dataset '{dataset_id}' toegevoegd!")
    print(f"📁 Configuratie opgeslagen in: {args.config}")
    print(f"\n🎯 Test je nieuwe dataset:")
    print(f"   python dataset_manager.py --setup {dataset_id}")
    print(f"\n📋 Of bekijk alle datasets:")
    print(f"   python dataset_manager.py --list")

if __name__ == "__main__":
    main()
